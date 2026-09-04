# Peak and training_compute_budget should be same units

peak = 121
cost = 0.9776    # Hourly rate of the GPU 
training_compute_budget = 1.144 * (10**6)
optimizer_steps = 3726
mfu_assumed = 0.10
observed_time_taken_to_complete_one_optimizer_step = None

compute_per_optimizer_step = training_compute_budget / optimizer_steps

est_time_to_train = (training_compute_budget) / (peak * mfu_assumed * 3600)
est_time_for_one_optimizer_step = (compute_per_optimizer_step) / (peak * mfu_assumed)

est_cost = cost * est_time_to_train

print(f"Compute per optimizer step: {compute_per_optimizer_step}")
print(f"Est time to train: {est_time_to_train:.3f} hrs")
print(f"Est time for 1 optimizer step: {est_time_for_one_optimizer_step:.3f} secs")
print(f"Est cost: ${est_cost}")


if observed_time_taken_to_complete_one_optimizer_step is not None:

    observed_perf = compute_per_optimizer_step / observed_time_taken_to_complete_one_optimizer_step

    observed_mfu = (observed_perf / peak) 

    print(f"Observed FLOP: {observed_perf}")
    print(f"Observed MFU: {observed_mfu}")