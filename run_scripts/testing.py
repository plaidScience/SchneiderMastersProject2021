import networks.PIX2PIX_DISC as pix2pix
import networks.UNET_GEN as unet

disc = pix2pix.PIX2PIX_DISC("./OUTPUT/TEST/_Disc")
gen = unet.UNET_GENERATOR("./OUTPUT/TEST/_Gen")
input("Waiting")
disc.summary()
gen.summary()
