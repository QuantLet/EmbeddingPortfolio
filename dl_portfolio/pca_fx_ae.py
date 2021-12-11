from dl_portfolio.run import run_ae
from dl_portfolio.logger import LOGGER
from joblib import parallel_backend, Parallel, delayed

LOG_DIR = 'dl_portfolio/log_fx_AE'

# 19,20,21,22,23,24,25,26,27,28,29,30

if __name__ == "__main__":
    import argparse
    from dl_portfolio.config import ae_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--n",
                        default=1,
                        type=int,
                        help="Number of experience to run")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Seed")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="increase output verbosity")
    parser.add_argument("--n_jobs",
                        default=1,
                        type=int,
                        help="Number of parallel jobs")
    parser.add_argument("--seeds",
                        nargs="+",
                        default=None,
                        help="List of seeds to run experiments")
    args = parser.parse_args()
    if args.seeds:
        if args.n_jobs == 1:
            for i, seed in enumerate(args.seeds):
                run_ae(ae_config, seed=int(seed))
        else:
            with parallel_backend("threading", n_jobs=args.n_jobs):
                Parallel()(
                    delayed(run_ae)(ae_config, seed=int(seed)) for seed in args.seeds
                )

    else:
        if args.n_jobs == 1:
            for i in range(args.n):
                LOGGER.info(f'Starting experiment {i+1} out of {args.n} experiments')
                if args.seed:
                    run_ae(ae_config, seed=args.seed)
                else:
                    run_ae(ae_config, seed=i)
                LOGGER.info(f'Experiment {i+1} finished')
                LOGGER.inof(f'{args.n - i - 1} experiments to go')
        else:
            if args.seed:
                with parallel_backend("threading", n_jobs=args.n_jobs):
                    Parallel()(
                        delayed(run_ae)(ae_config, seed=args.seed) for i in range(args.n)
                    )
            else:
                with parallel_backend("threading", n_jobs=args.n_jobs):
                    Parallel()(
                        delayed(run_ae)(ae_config, seed=seed) for seed in range(args.n)
                    )