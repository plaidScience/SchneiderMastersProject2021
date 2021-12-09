import sys

i = int(input("Input the ID of the Run Script to Run: "))

if i == -1:
    print("Running Tester Script")
    import run_scripts.testing as to_run
elif i == -2:
    import networks.CYCLEGAN as to_run

elif i == -3:
    import run_scripts.sg_base.celebA_TESTING_STARGAN as to_run
    print("Running test on STARGAN on celebA local Download")
elif i == -4:
    import run_scripts.sc_multicycle.celebA_TESTING_STARGAN_multicycle as to_run
    print("Running test on STARGAN with Multi-Cyclical on celebA local Download")


elif i == 1:
    import run_scripts.resnet_cycleGAN as to_run
    print("Running CycleGAN with RESNET Generator")


elif i == 2:
    import run_scripts.unet_cycleGAN as to_run
    print("Running CycleGAN with UNET Generator")
elif i == 3:
    import run_scripts.sg_base.celebA_local_STARGAN as to_run
    print("Running STARGAN on celebA local Download")
elif i == 4:
    import run_scripts.sc_multicycle.celebA_local_STARGAN_multicycle as to_run
    print("Running STARGAN with Mutli-Cyclical Loss on celebA local Download")

if i >= 3:
    reload = input("Reload from recent checkpointed model? [y/N]".startswith('y'))
    to_run.main(reload)
else:
    to_run.main()
