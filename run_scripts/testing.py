import networks.PIX2PIX_DISC as pix2pix
import networks.UNET_GEN as unet
import networks.STARGAN_GENERATOR as stargan_gen
import networks.STARGAN_DISCRIMINATOR as stargan_disc

def main():
    inp_shape = [256, 256, 3]
    disc = pix2pix.PIX2PIX_DISC(inp_shape, "./OUTPUT/TEST/_Disc")
    gen = unet.UNET_GENERATOR(inp_shape, "./OUTPUT/TEST/_Gen")
    input("Waiting")
    disc.summary()
    gen.summary()
    input("Stargan Gens")
    inp_shape = [256, 256, 3]
    n_classes = 3
    sgen = stargan_gen.RESNET_GENERATOR(inp_shape, n_classes, "./OUTPUT/TEST/_StarGAN_Gen")
    sdisc = stargan_disc.PIX2PIX_DISC(inp_shape, n_classes, "./OUTPUT/TEST/_StarGAN_Disc")
    input("Waiting")
    sgen.summary()
    sdisc.summary()


if __name__ == '__main__':
    main()
