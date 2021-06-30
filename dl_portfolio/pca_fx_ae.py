from dl_portfolio.run import run

LOG_DIR = 'dl_portfolio/log_fx_AE'


if __name__ == "__main__":
    from dl_portfolio.config import ae_config

    run(ae_config)