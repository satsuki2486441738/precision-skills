"""Portable four-GPU NCCL + optional native TRL SFTTrainer smoke; no downloads.

Run: python remote_four_gpu_probe.py --output-dir /user/owned/new-output-parent
Creates a fresh compare-four-gpu-UTC folder and launches real torchrun, four ranks.
Torch/CUDA/NCCL are mandatory. Optional native SFT needs transformers==4.57.1,
trl==0.24.0, accelerate==1.10.1, datasets==4.2.0, tokenizers and safetensors.
The SFT model is a seeded random tiny Qwen2, not pretrained or RL training.
Only the fresh run receives files/caches. The launcher has a 285-second timeout.
No reference gradient comparison is claimed; actual collated batches are captured.
"""
import argparse, datetime, hashlib, importlib, json, os, pathlib, signal
import subprocess, sys, threading, time, traceback

def dump(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def digest(raw):return hashlib.sha256(raw).hexdigest()

def launch(args):
    base=pathlib.Path(args.output_dir).expanduser().resolve()
    base.mkdir(parents=True,exist_ok=True)
    run=base/datetime.datetime.now(datetime.timezone.utc).strftime('compare-four-gpu-%Y%m%dT%H%M%SZ')
    run.mkdir(exist_ok=False)
    for name in ['tmp','cache','trainer_output']: (run/name).mkdir()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',WANDB_MODE='disabled',
        HF_HOME=str(run/'cache/hf'),HF_DATASETS_CACHE=str(run/'cache/datasets'),
        XDG_CACHE_HOME=str(run/'cache'),TRITON_CACHE_DIR=str(run/'cache/triton'),
        TORCH_HOME=str(run/'cache/torch'),TMPDIR=str(run/'tmp'),
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1',OMP_NUM_THREADS='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
             '--nproc_per_node=4',str(pathlib.Path(__file__).resolve()),
             '--output-dir',str(run),'--worker']
    dump(run/'launch.json',dict(command=command,script_sha256=digest(pathlib.Path(__file__).read_bytes()),
        limits=dict(timeout_seconds=285,world_size=4,model_weight_downloads=False),
        intended_stages=['NCCL init/all_reduce','optional native SFTTrainer.train(max_steps=1)']))
    started=time.time();timed_out=False
    with (run/'torchrun.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                                 cwd=run,start_new_session=True)
        try: code=process.wait(timeout=285)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    ranks=[]
    for rank in range(4):
        path=run/f'rank-{rank}.json'
        ranks.append(json.loads(path.read_text()) if path.exists() else dict(rank=rank,status='missing'))
    nccl=all(x.get('nccl',{}).get('status')=='completed' for x in ranks)
    sft=all(x.get('status')=='completed' and x.get('sft',{}).get('status')=='completed' for x in ranks)
    grad_hashes=[x.get('sft',{}).get('gradient',{}).get('sha256') for x in ranks]
    after_hashes=[x.get('sft',{}).get('after_parameters_sha256') for x in ranks]
    summary=dict(status='completed' if nccl and sft and code==0 else 'partial' if nccl else 'failed',
        run=str(run),returncode=code,timed_out=timed_out,elapsed_seconds=time.time()-started,
        world_size=4,nccl_four_rank_completed=nccl,native_sft_trainer_completed=sft,
        native_rl_training=False,matched_reference_executed=False,thresholds=None,
        cross_rank_gradient_hash_equal=sft and len(set(grad_hashes))==1,
        cross_rank_updated_parameter_hash_equal=sft and len(set(after_hashes))==1,ranks=ranks)
    dump(run/'summary.json',summary)
    print(json.dumps(dict(run=str(run),summary=str(run/'summary.json'),status=summary['status'],
                         nccl_four_rank_completed=nccl,native_sft_trainer_completed=sft),indent=2))
    return 0 if nccl and sft and code==0 and not timed_out else 1

def worker(args):
    run=pathlib.Path(args.output_dir).resolve()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    path=run/f'rank-{rank}.json'
    record=dict(rank=rank,local_rank=local,world_size=int(os.environ['WORLD_SIZE']),
                status='initializing',nccl=dict(status='not_started'),sft=dict(status='not_started'))
    dump(path,record)
    def expired():
        record['status']='timeout';record['timeout_seconds']=250
        dump(path,record);os._exit(124)
    watchdog=threading.Timer(250,expired);watchdog.daemon=True;watchdog.start()
    dist=None
    try:
        import torch
        import torch.distributed as dist
        torch.set_num_threads(1);torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert record['world_size']==4 and torch.cuda.device_count()>=4
        dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=60))
        value=torch.tensor([float(rank)],dtype=torch.float64,device=f'cuda:{local}')
        dist.all_reduce(value,op=dist.ReduceOp.SUM);torch.cuda.synchronize()
        assert float(value)==6.0
        record['runtime']=dict(python=sys.version,torch=torch.__version__,torch_path=torch.__file__,
            cuda=torch.version.cuda,nccl_version=torch.cuda.nccl.version(),gpu=torch.cuda.get_device_name(local),
            capability=list(torch.cuda.get_device_capability(local)),tf32=False)
        record['nccl']=dict(status='completed',backend=dist.get_backend(),actual_world_size=dist.get_world_size(),
            all_reduce_input=float(rank),all_reduce_result=float(value),expected=6.0)
        record['status']='nccl_completed';dump(path,record)
        print(json.dumps(dict(rank=rank,stage='NCCL',status='completed')),flush=True)
        imported={};error=None
        try:
            for name in ['transformers','trl','accelerate','datasets','tokenizers','safetensors']:
                module=importlib.import_module(name)
                imported[name]=dict(version=getattr(module,'__version__',None),path=getattr(module,'__file__',None))
            from transformers import AutoModelForCausalLM,PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
            from trl import SFTConfig,SFTTrainer
            from datasets import Dataset
            from tokenizers import Tokenizer,models,pre_tokenizers
        except Exception: error=traceback.format_exc()
        available=[None]*4;dist.all_gather_object(available,dict(rank=rank,error=error))
        record['optional_imports']=imported
        if any(item['error'] for item in available):
            record['sft']=dict(status='unavailable',prerequisite_errors=available)
            record['status']='completed_nccl_only';dump(path,record)
            return 0
        seed=20261019;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        checkpoint=run/'initial_tiny_qwen2'
        if rank==0:
            config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
                bos_token_id=2,eos_token_id=3,pad_token_id=1,tie_word_embeddings=True,attention_dropout=0.)
            initial=Qwen2ForCausalLM(config);initial.save_pretrained(checkpoint,safe_serialization=True)
            vocab={'<unk>':0,'<pad>':1,'<bos>':2,'<eos>':3,**{f't{i}':i for i in range(4,64)}}
            raw_tokenizer=Tokenizer(models.WordLevel(vocab=vocab,unk_token='<unk>'))
            raw_tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,unk_token='<unk>',
                pad_token='<pad>',bos_token='<bos>',eos_token='<eos>',model_max_length=16)
            tokenizer.save_pretrained(checkpoint);del initial
        dist.barrier()
        model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='eager')
        model.config.use_cache=False
        tokenizer=PreTrainedTokenizerFast.from_pretrained(checkpoint,local_files_only=True)
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        def tensor_hash(values):
            hasher=hashlib.sha256()
            for name,tensor in values:
                hasher.update(name.encode());hasher.update(str(tuple(tensor.shape)).encode())
                hasher.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
            return hasher.hexdigest()
        initial_hash=tensor_hash(before.items())
        rows=[]
        for sample in range(8):
            ids=[2]+[4+((sample*7+j)%58) for j in range(14)]+[3]
            rows.append(dict(input_ids=ids,attention_mask=[1]*16,labels=ids.copy()))
        batches=[];gradient={}
        class ObservedSFTTrainer(SFTTrainer):
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                if not batches:
                    batches.append({key:value.detach().cpu().tolist() for key,value in inputs.items()
                                    if torch.is_tensor(value) and key in ['input_ids','labels','attention_mask','position_ids']})
                    batches[-1]['num_items_in_batch']=float(num_items_in_batch) if num_items_in_batch is not None else None
                return super().compute_loss(model,inputs,return_outputs=return_outputs,num_items_in_batch=num_items_in_batch)
        settings=dict(output_dir=str(run/'trainer_output'),max_steps=1,per_device_train_batch_size=2,
            gradient_accumulation_steps=1,learning_rate=.01,optim='sgd',weight_decay=0.,
            lr_scheduler_type='constant',warmup_steps=0,max_grad_norm=0.,seed=seed,data_seed=seed,
            bf16=False,fp16=False,gradient_checkpointing=False,packing=False,max_length=16,
            completion_only_loss=False,assistant_only_loss=False,report_to=[],save_strategy='no',
            eval_strategy='no',logging_strategy='steps',logging_steps=1,dataloader_num_workers=0,
            dataloader_drop_last=True,dataloader_pin_memory=False,ddp_find_unused_parameters=False,
            remove_unused_columns=False,disable_tqdm=True)
        trainer=ObservedSFTTrainer(model=model,args=SFTConfig(**settings),
            train_dataset=Dataset.from_list(rows),processing_class=tokenizer)
        trainer.create_optimizer()
        def before_optimizer_step(optimizer,*unused):
            items=[];per_tensor=[];total_sq=0.;maximum=0.;present=0
            for name,parameter in model.named_parameters():
                if parameter.grad is None:
                    per_tensor.append(dict(name=name,present=False,elements=parameter.numel()));continue
                value=parameter.grad.detach().float().cpu();items.append((name,value))
                norm=float(value.double().norm());maximum=max(maximum,float(value.abs().max()))
                total_sq+=norm*norm;present+=value.numel()
                per_tensor.append(dict(name=name,present=True,elements=value.numel(),l2=norm,max_abs=float(value.abs().max())))
            gradient.update(sha256=tensor_hash(items),elements=present,l2=total_sq**.5,max_abs=maximum,
                per_tensor=per_tensor,capture='native optimizer pre-step hook after DDP synchronization; clipping disabled')
        trainer.optimizer.register_step_pre_hook(before_optimizer_step)
        record['sft']=dict(status='running',framework='TRL SFTTrainer.train',seed=seed,
            model_origin='seeded random tiny Qwen2; not pretrained',initial_parameter_sha256=initial_hash,
            checkpoint_files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in checkpoint.iterdir() if p.is_file()},
            settings=settings,dataset_sha256=digest(json.dumps(rows,sort_keys=True).encode()))
        dump(path,record)
        trained=trainer.train()
        after=tensor_hash(model.named_parameters());delta=[]
        for name,parameter in model.named_parameters():delta.append((parameter.detach().float().cpu()-before[name]).reshape(-1))
        updates=torch.cat(delta).double()
        assert trainer.state.global_step==1 and trainer.accelerator.num_processes==4 and gradient
        record['sft'].update(status='completed',native_global_step=trainer.state.global_step,
            native_loss=float(trained.training_loss),native_metrics=trained.metrics,
            accelerator_world_size=trainer.accelerator.num_processes,actual_collated_batches=batches,
            gradient=gradient,after_parameters_sha256=after,
            stored_update=dict(elements=updates.numel(),nonzero=int(torch.count_nonzero(updates)),
                l2=float(updates.norm()),rmse=float(updates.square().mean().sqrt()),max_abs=float(updates.abs().max())),
            optimizer_state=trainer.optimizer.state_dict()['param_groups'],native_rl_training=False,
            matched_reference_executed=False,thresholds=None)
        record['status']='completed';dump(path,record);dist.barrier()
        print(json.dumps(dict(rank=rank,stage='native_SFTTrainer',status='completed',loss=float(trained.training_loss))),flush=True)
        return 0
    except Exception:
        record['status']='failed';record['error']=traceback.format_exc();dump(path,record)
        print(record['error'],flush=True);return 1
    finally:
        watchdog.cancel()
        if dist is not None and dist.is_initialized():
            try:dist.destroy_process_group()
            except Exception:pass

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    arguments=parser.parse_args()
    raise SystemExit(worker(arguments) if arguments.worker else launch(arguments))
