import sys

if len(sys.argv) > 1:
    i = int(sys.argv[1])
else:
    i = int(input("Input the ID of the Run Script to Run: "))
if i == -1:
    import run_scripts.testing
if i == -2:
        import networks.CYCLEGAN
        networks.CYCLEGAN.main()

if i == 1:
    import run_scripts.resnet_cycleGAN

if i == 2:
    import run_scripts.unet_cycleGAN
