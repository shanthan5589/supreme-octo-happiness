# Peak and training_compute_budget should be same units

peak = 362.05
cost = 1.861    # Hourly rate of the GPU 
training_compute_budget = 1.144 * (10**6)
optimizer_steps = 3726

mfu_assumed = 0.10
assumed_time_to_compute_eval_loss = 20     # 128 (64 + 64)

observed_time_taken_to_complete_one_optimizer_step = 17.16
observed_time_to_compute_eval_loss = 23.83

eval_interval = 10

total_eval_batches = (optimizer_steps // eval_interval) if optimizer_steps % eval_interval == 0 else (optimizer_steps // eval_interval) + 1


compute_per_optimizer_step = training_compute_budget / optimizer_steps

est_time_to_train = (training_compute_budget) / (peak * mfu_assumed * 3600)
est_time_for_one_optimizer_step = (compute_per_optimizer_step) / (peak * mfu_assumed)

total_time = est_time_to_train + ((total_eval_batches * assumed_time_to_compute_eval_loss) / 3600)

est_cost = total_time * cost

print(f"Compute per optimizer step: {compute_per_optimizer_step}")
print(f"Est time to train: {total_time:.3f} hrs")
print(f"Est time for 1 optimizer step: {est_time_for_one_optimizer_step:.3f} secs")
print(f"Est cost: ${est_cost}")


if observed_time_taken_to_complete_one_optimizer_step is not None:

    observed_perf = compute_per_optimizer_step / observed_time_taken_to_complete_one_optimizer_step

    observed_mfu = (observed_perf / peak) 

    calc_time_to_train = (training_compute_budget) / (peak * observed_mfu * 3600)
    calc_time_for_one_optimizer_step = (compute_per_optimizer_step) / (peak * observed_mfu)


    time_spent_on_eval_batches = (total_eval_batches * observed_time_to_compute_eval_loss) / 3600

    total_time = calc_time_to_train + time_spent_on_eval_batches 

    calc_cost = total_time * cost


    print(f"Observed FLOP: {observed_perf}")
    print(f"Observed MFU: {observed_mfu}")

    print(f"Calculated time to train: {total_time} hrs")
    print(f"Calculated time for 1 optimizer step: {calc_time_for_one_optimizer_step} secs")
    print(f"Calculated cost: ${calc_cost}")