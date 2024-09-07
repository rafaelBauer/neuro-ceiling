import wandb

api = wandb.Api()
run = api.run("rafael-bauer/StackCubesInd/5d748xpb")
run.summary["model_name"] = "ceiling_10_pretrain_policy_0"
run.summary["model_version"] = "ceiling_10_pretrain_policy_0:v0"
run.summary.update()
