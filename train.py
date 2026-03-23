# /home/610-sty/compare/poe2clp/train.py
import os
import pytorch_lightning as pl
from argparse import ArgumentParser

from models.model import STYText2ImageModel
from utils.tokenizer import ChineseTokenizer
from utils.collate import custom_collate
from data.universal_datamodule import DataModuleCustom

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

# 禁用 Tokenizer 的并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # 1. 初始化分词器
    tokenizer = ChineseTokenizer()

    # 2. 配置参数解析器
    parser = ArgumentParser()
    
    # ✅ 自动添加 DataModule 的参数，包括 --webdataset_base_urls, --batch_size, --num_workers
    parser = DataModuleCustom.add_data_specific_args(parser)  
    
    # 3. 训练超参数
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-5) 
    parser.add_argument('--use_learnable_extractor', action='store_true',
                        help="是否使用可训练的短语提取器")
    parser.add_argument('--debug', action='store_true',
                        help="启用调试模式（限制数据量）")
    parser.add_argument('--fast_dev_run', action='store_true',
                        help="快速开发运行模式")
    parser.add_argument('--devices', type=int, default=1,
                        help="使用的 GPU 数量")

    args = parser.parse_args()

    # 4. 构造数据模块
    datamodule = DataModuleCustom(
        args=args,
        tokenizer=tokenizer,
        collate_fn=custom_collate,
        use_worker_init_fn=True
    )

    # 5. 构造 Lightning 模型
    model = STYText2ImageModel(
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        freeze_unet_epochs=0,
        use_learnable_extractor=args.use_learnable_extractor
    )

    # 6. 准备工作目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/phrase_attention_maps", exist_ok=True)

    # 7. 配置回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="sty-p2c-{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=10,
        save_last=True,
        save_on_train_epoch_end=True,
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = CSVLogger("logs", name="sty_t2i_v4.4")

    # 8. 构造 Trainer
    trainer_args = dict(
        accelerator='gpu',
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        # ✅ 【核心修复】根据你的环境报错提示，将 'bf16-mixed' 改为 'bf16'
        precision="bf16", 
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        strategy="auto" if args.devices == 1 else "ddp",
    )

    if args.debug:
        trainer_args.update({"limit_train_batches": 0.05, "log_every_n_steps": 1})
    if args.fast_dev_run:
        trainer_args["fast_dev_run"] = True

    trainer = pl.Trainer(**trainer_args)

    # 9. 打印训练摘要
    print("\n" + "="*50)
    print("🚀 POE2CLP Training Start")
    print(f"   - Epochs: {args.max_epochs}")
    print(f"   - Learning Rate: {args.learning_rate}")
    print(f"   - Precision: {trainer_args['precision']}")
    print(f"   - Learnable Extractor: {args.use_learnable_extractor}")
    print(f"   - GPU Devices: {args.devices}")
    print("="*50 + "\n")

    # 10. 启动训练
    try:
        trainer.fit(model, datamodule)
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise e

    # 11. 保存最终模型
    final_ckpt_path = os.path.join("checkpoints", "final_model.ckpt")
    trainer.save_checkpoint(final_ckpt_path)
    print(f"💾 Final model checkpoint saved to: {final_ckpt_path}")

if __name__ == "__main__":
    main()