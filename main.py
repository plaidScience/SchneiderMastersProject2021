import sys

i = int(input("Input the ID of the Run Script to Run: "))

if i == -1:
    print("Running Tester Script")
    import run_scripts.testing as to_run
if i == -2:
    import networks.CYCLEGAN as to_run

if i == 1:
    import run_scripts.resnet_cycleGAN as to_run
    print("Running CycleGAN with RESNET Generator")


if i == 2:
    import run_scripts.unet_cycleGAN as to_run
    print("Running CycleGAN with UNET Generator")

to_run.main()
