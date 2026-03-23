# train.py
import os
import pytorch_lightning as pl
from argparse import ArgumentParser

from models.model import STYText2ImageModel
from utils.tokenizer import ChineseTokenizer
from utils.collate import custom_collate
from data.universal_datamodule import DataModuleCustom

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # 初始化 tokenizer
    tokenizer = ChineseTokenizer()

    # 配置参数解析器
    parser = ArgumentParser()
    
    # 添加 DataModuleCustom 的参数
    parser = DataModuleCustom.add_data_specific_args(parser)  
    
    # 添加模型训练相关参数
    parser.add_argument('--max_epochs', type=int, default=100)  # 增加到 100
    parser.add_argument('--learning_rate', type=float, default=5e-5)  # ✅ 从 1e-4 降到 5e-5
    parser.add_argument('--use_learnable_extractor', action='store_true',
                        help="是否使用可训练的关键词提取器（替代 jieba）")
    parser.add_argument('--debug', action='store_true',
                        help="启用调试模式（更详细的日志）")
    parser.add_argument('--fast_dev_run', action='store_true',
                        help="快速开发运行（只跑几个batch）")
    parser.add_argument('--devices', type=int, default=1,  # 添加设备选择
                        help="使用的 GPU 数量")

    args = parser.parse_args()

    # 构造数据模块
    datamodule = DataModuleCustom(
        args=args,
        tokenizer=tokenizer,
        collate_fn=custom_collate,
        use_worker_init_fn=True
    )

    # 构造模型
    model = STYText2ImageModel(
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        freeze_unet_epochs=0,
        use_learnable_extractor=args.use_learnable_extractor
    )

    # 创建目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/phrase_attention_maps", exist_ok=True)

    # Checkpoint 回调 - 调整保存频率
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="sty-epoch{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=10,          # 每10个epoch保存一次（减少存储压力）
        save_last=True,
        save_on_train_epoch_end=True,
        verbose=False               # 减少日志输出
    )

    # 学习率监控回调
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 日志记录器
    logger = CSVLogger("logs", name="sty_t2i")

    # 构造 trainer
    trainer_args = dict(
        accelerator='gpu',
        devices=args.devices,       # 使用指定的设备数
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        precision="16",             # 混合精度训练
        gradient_clip_val=1.0,
        log_every_n_steps=10,       # 适度的日志频率
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
    )

    # 调试模式配置
    if args.debug:
        trainer_args.update({
            "limit_train_batches": 0.05,  # 只使用5%的数据进行调试
            "limit_val_batches": 0.05,
            "log_every_n_steps": 1,
        })
        print("⚠️  Debug mode enabled - using limited data")

    # 快速开发运行
    if args.fast_dev_run:
        trainer_args["fast_dev_run"] = True
        print("⚡ Fast dev run enabled - running quick test")

    trainer = pl.Trainer(**trainer_args)

    # 显示训练配置
    print("\n" + "="*50)
    print("Training Configuration:")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")  # ✅ 现在会显示 5e-5
    print(f"Use learnable extractor: {args.use_learnable_extractor}")
    print(f"Devices: {args.devices}")
    print("="*50 + "\n")

    # 开始训练
    print("🚀 Starting training...")
    trainer.fit(model, datamodule)
    print("✅ Training completed!")

    # 保存最终模型
    final_ckpt_path = os.path.join("checkpoints", "final.ckpt")
    trainer.save_checkpoint(final_ckpt_path)
    print(f"💾 Final model saved to: {final_ckpt_path}")

if __name__ == "__main__":
    main()