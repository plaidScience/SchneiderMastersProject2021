import sys

if len(sys.argv) > 1:
    i = int(sys.argv[1])
else:
    i = int(input("Input the ID of the Run Script to Run: "))
if i == -1:
    import run_scripts.testing
if i == -2:
    import networks.CYCLEGAN as to_run

if i == 1:
    import run_scripts.resnet_cycleGAN as to_run

if i == 2:
    import run_scripts.unet_cycleGAN as to_run

to_run.main()
