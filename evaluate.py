if 'qual' in input("Qualitative or [Quantitative]: ").lower():
    import run_scripts.evaluation.celeba_evaluate_qualitative as to_run  
    to_run.main()  
else:
    import run_scripts.evaluation.celeba_evaluate as to_run
    to_run.main(use_lpips=input("Use External LPIPS? [y/N]: ").startswith('y'))