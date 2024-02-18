!#/bin/bash

# Set if current system has CUDA-enabled GPU
cuda_enabled = 0

# num_trials = 10
# num_warmup = 3

num_benchmarks = 0

for nx in $(seq 4 4); do
    for ny in $(seq 1 12); do
        ny=$((2**ny))
        for mode in "RATE" "PREC" "ACC" "REV"; do

            # Create ZFP compression mode flag parameter arrays
            if [ "$mode" == "RATE" ]; then
                comp_param_flag="--rate="
                comp_param_arr=$(seq 1 32)
            elif [ "$mode" == "PREC" ]; then
                comp_param_flag="--precision="
                comp_param_arr=$(seq 1 32)
            elif [ "$mode" == "ACC" ]; then
                comp_param_flag="--tolerance="
                comp_param_arr=$(seq 1 32)
            elif [ "$mode" == "REV" ]; then
                comp_param_flag=""
                comp_param_arr=("")
            fi

            for comp_param in $comp_param_arr; do

                for exec in "CPU" "OMP" "CUDA"; do

                    # Create array of OpenMP threads to use
                    if [ "$exec" == "OMP" ]; then
                        num_threads_arr=$(seq 1 16)
                    else
                        num_threads_arr=(1)
                    fi

                    for num_threads in $num_threads_arr; do

                        # Don't run CUDA tests if not available
                        if [ "$exec" == "CUDA" ] && [ ! $cuda_enabled ]; then
                            break
                        fi

                        cmd="./zfp_bench --nx="${nx}" --ny="${ny}" --mode=$mode --exec=$exec --threads=$num_threads $comp_param_flag$comp_param"
                        echo "$cmd"
                        $cmd

                        num_benchmarks=$((num_benchmarks+1))
                    done
                done
            done
        done
    done
done

echo "Benchmarking Suite Done."
echo "$num_benchmarks benchmarks ran."