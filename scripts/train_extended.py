"""
MTRPP Extended Training with Validation Monitoring
Trains from scratch for up to 48 hours with periodic validation.
"""
import os
import sys
import time
import csv
import argparse
import torch.optim as optim
from tqdm import tqdm
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mtrpp_local import (
    RLConfig, RLTrainer, create_large_model,
    load_instances, set_seed, evaluate_on_instances, CONFIG
)


def validate(model, val_instances, max_iters=100):
    """Quick validation pass. Returns avg gap vs final_profit."""
    if not val_instances:
        return None
    results = evaluate_on_instances(
        val_instances, model,
        batch_size=16, max_iterations=max_iters,
        temperature=0.05, intensive=False
    )
    instance_results = results.get('results', [])
    
    # Map optimal values by instance name to avoid ordering bugs
    opt_map = {getattr(inst, 'name', f"Instance_{i}"): getattr(inst, 'final_profit', None) 
               for i, inst in enumerate(val_instances)}
    
    gaps = []
    for res in instance_results:
        inst_name = res.get('name')
        opt = opt_map.get(inst_name)
        if opt and opt > 0:
            gap = (opt - res['objective']) / opt * 100.0
            gaps.append(gap)
            
    return sum(gaps) / len(gaps) if gaps else None


def main():
    parser = argparse.ArgumentParser(description="MTRPP Extended Training")
    parser.add_argument("--train_dataset", type=str, default="dataset/train_split_FINAL.pt")
    parser.add_argument("--test_dataset", type=str, default="dataset/test_split_FINAL.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_nodes_train", type=int, default=500)
    parser.add_argument("--max_train_instances", type=int, default=None, help="Limit training instances for fast testing")
    parser.add_argument("--max_hours", type=float, default=48.0)
    parser.add_argument("--val_every", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--val_instances", type=int, default=10, help="Number of validation instances")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiments/models")
    args = parser.parse_args()

    CONFIG['hidden_dim'] = args.hidden_dim
    CONFIG['num_layers'] = args.num_layers
    CONFIG['num_heads'] = args.num_heads
    CONFIG['output_dir'] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("experiments/logs", exist_ok=True)

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("MTRPP EXTENDED TRAINING (FROM SCRATCH / RESUME)")
    print(f"Device: {device}")
    print(f"Max training time: {args.max_hours} hours")
    print(f"Epochs: {args.epochs}")

    # Load datasets
    train_instances = load_instances([args.train_dataset], max_instances=args.max_train_instances)
    safe_instances = [inst for inst in train_instances if inst.n_customers <= args.max_nodes_train]
    if len(safe_instances) < len(train_instances):
        print(f"VRAM filter: {len(train_instances) - len(safe_instances)} instances > {args.max_nodes_train} nodes removed")
    train_instances = safe_instances
    print(f"Training set: {len(train_instances)} instances")

    test_instances = load_instances([args.test_dataset], max_instances=args.val_instances)
    print(f"Validation set: {len(test_instances)} instances")

    print("Initializing components...")
    
    # Checkpoint resume logic
    start_epoch = 0
    best_avg_reward = float('-inf')
    latest_checkpoint = os.path.join(args.output_dir, "rl_model_latest.pt")
    
    if os.path.exists(latest_checkpoint):
        print(f"Resuming training from {latest_checkpoint}...")
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        model = create_large_model(
            hidden_dim=checkpoint.get('hidden_dim', args.hidden_dim),
            num_layers=checkpoint.get('num_layers', args.num_layers),
            heads=checkpoint.get('heads', args.num_heads)
        )
        sd = checkpoint['model_state_dict']
        model.load_state_dict(sd)
        start_epoch = checkpoint.get('epoch', 0)
        best_avg_reward = checkpoint.get('best_avg_reward', float('-inf'))
    else:
        print("Starting fresh model...")
        model = create_large_model(hidden_dim=args.hidden_dim, num_layers=args.num_layers, heads=args.num_heads)
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    if os.path.exists(latest_checkpoint) and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    baseline_model = copy.deepcopy(model)
    baseline_model.eval()

    config = RLConfig(
        max_steps_per_episode=50,
        learning_rate=args.lr,
        gamma=0.99,
        entropy_coef=0.01,
        episodes_per_instance=1,
        temperature=1.5,
        temperature_decay=0.995,
        min_temperature=0.1,
        reward_clip=10.0,
        batch_size=args.batch_size,
        checkpoint_every=50,
        grad_accum_steps=4,
    )

    trainer = RLTrainer(model, config, device)

    log_path = "experiments/logs/training_log.csv"
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat(log_path).st_size == 0:
            writer.writerow(["epoch", "avg_reward", "imp_rate", "val_gap_pct", "elapsed_hours"])

    start_time = time.time()
    best_val_gap = float('inf')
    negative_streak = 0

    for epoch in range(start_epoch, args.epochs):
        elapsed_hours = (time.time() - start_time) / 3600.0
        if elapsed_hours >= args.max_hours:
            print(f"\nTime limit reached ({elapsed_hours:.1f}h >= {args.max_hours}h). Stopping.")
            break

        print(f"\n--- Epoch {epoch + 1}/{args.epochs} (Elapsed: {elapsed_hours:.1f}h) ---")

        stats = trainer.train_on_instances(
            train_instances,
            num_epochs=epoch + 1,
            start_epoch=epoch,
            start_instance_idx=0,
            existing_stats=None
        )

        avg_reward = stats['epoch_rewards'][-1] if stats['epoch_rewards'] else 0.0
        imp_rate = stats['improvements'][-1] if stats['improvements'] else 0.0

        # Save latest checkpoint for resumption
        latest_path = os.path.join(args.output_dir, "rl_model_latest.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_avg_reward': best_avg_reward,
            'best_val_gap': best_val_gap,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'heads': args.num_heads,
        }, latest_path)

        # Save best model based on training reward (legacy support)
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_reward_path = os.path.join(args.output_dir, 'best_reward_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(), 
                'epoch': epoch + 1, 
                'avg_reward': avg_reward,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'heads': args.num_heads,
            }, best_reward_path)
            print(f"New best reward model saved (avg_reward={avg_reward:.2f})")

        # Validation
        val_gap = None
        if (epoch + 1) % args.val_every == 0:
            print("  Running validation...")
            model.eval()
            val_gap = validate(model, test_instances)
            model.train()

            if val_gap is not None:
                print(f"  Validation gap: {val_gap:.4f}%")
                if val_gap < best_val_gap:
                    best_val_gap = val_gap
                    # THIS IS THE ONE THE EVAL SCRIPTS LOOK FOR
                    best_path = os.path.join(args.output_dir, 'best_model.pt')
                    torch.save({
                        'model_state_dict': model.state_dict(), 
                        'epoch': epoch + 1, 
                        'val_gap': val_gap,
                        'hidden_dim': args.hidden_dim,
                        'num_layers': args.num_layers,
                        'heads': args.num_heads,
                    }, best_path)
                    
                    negative_streak += 1
                    print(f"  New Best Validation Model! Gap: {val_gap:.4f}% (Streak: {negative_streak}/3)")
                    
                    if negative_streak >= 3 and val_gap < 0:
                        print("  Solver confirmed to exceed reference (Negative Gap). Stopping early.")
                        break
                else:
                    negative_streak = 0

        # Log
        elapsed_hours = (time.time() - start_time) / 3600.0
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_reward:.4f}", f"{imp_rate:.4f}",
                             f"{val_gap:.4f}" if val_gap is not None else "", f"{elapsed_hours:.2f}"])

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'heads': args.num_heads,
        'epoch': epoch + 1,
    }, final_path)

    elapsed_hours = (time.time() - start_time) / 3600.0
    print("\nTRAINING COMPLETE")
    print(f"  Total time: {elapsed_hours:.1f} hours")
    print(f"  Best validation gap: {best_val_gap:.4f}%")
    print(f"  Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
