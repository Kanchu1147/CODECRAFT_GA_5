# CODECRAFT_GA_5
 Neural Style Transfer
#InsideNST
How Do Neural Style Transfers Work?
Deep Learning made it possible to capture the content of one image and combine it with the style of another image. This technique is called Neural Style Transfer. But, how Neural Style Transfer works? In this blog post, we are going to look into the underlying mechanism of Neural Style Transfer (NST).
High-Level Intuition
https://miro.medium.com/v2/resize:fit:828/format:webp/0*bWkWBhe1HbidTTL6.png
(Ancient city of Persepolis)
+https://miro.medium.com/v2/resize:fit:828/format:webp/0*bWkWBhe1HbidTTL6.png
(The Starry Night by Van Gogh)
=https://miro.medium.com/v2/resize:fit:828/format:webp/0*bWkWBhe1HbidTTL6.png
(Persepolis in Van Gogh style)
Neural Style Transfer Overview

As we can see, the generated image is having the content of the Content image and style of the style image. It can be seen that the above result cannot be obtained simply by overlapping the images. Now, the million-dollar question remains, how we make sure that the generated image has the content of content image and style of style image? How we capture the content and style of respective images?
In order to answer the above questions, let’s look at what Convolutional Neural Networks(CNN) are actually learning.

What Convolutional Neural Network Captures ?
Look at the following image.

https://miro.medium.com/v2/resize:fit:828/format:webp/0*PW4Iu1EO9Kckn5lN.jpeg

Different Layers of Convolutional Neural Network

Now, at Layer 1 using 32 filters the network may capture simple patterns, say a straight line or a horizontal line which may not make sense to us but is of immense importance to the network, and slowly as we move down to Layer 2 which has 64 filters, the network starts to capture more and more complex features it might be a face of a dog or wheel of a car. This capturing of different simple and complex features is called feature representation.
Important thing to not here is that CNNs does not know what the image is, but they learn to encode what a particular image represents. This encoding nature of Convolutional Neural Networks can help us in Neural Style Transfer. Let’s dive a bit more deeper.

How Convolutional Neural Networks are used to capture Content and Style of images?
VGG19 network is used for Neural Style transfer. VGG-19 is a convolutional neural network that is trained on more than a million images from the ImageNet database. The network is 19 layers deep and trained on millions of images. Because of which it is able to detect high-level features in an image.
Now, this ‘encoding nature’ of CNN’s is the key in Neural Style Transfer. Firstly, we initialize a noisy image, which is going to be our output image(G). We then calculate how similar is this image to the content and style image at a particular layer in the network(VGG network). Since we want that our output image(G) should have the content of the content image(C) and style of style image(S) we calculate the loss of generated image(G) w.r.t to the respective content(C) and style(S) image.
Having the above intuition, let’s define our Content Loss and Style loss to randomly generated noisy image.

https://miro.medium.com/v2/resize:fit:828/format:webp/0*35-GmlSqciWqWaCC.png

Content Loss
Calculating content loss means how similar is the randomly generated noisy image(G) to the content image(C).In order to calculate content loss :

Assume that we choose a hidden layer (L) in a pre-trained network(VGG network) to compute the loss.Therefore, let P and F be the original image and the image that is generated.And, F[l] and P[l] be feature representation of the respective images in layer L.Now,the content loss is defined as follows:

https://miro.medium.com/v2/resize:fit:828/format:webp/0*PJK8-P3tBWrUV1q1.png

This concludes the Content loss function.

Style Loss
Before calculating style loss, let’s see what is the meaning of “style of a image” or how we capture style of an image.

How we capture style of an image ?

https://miro.medium.com/v2/resize:fit:352/format:webp/0*dyVKNRn36XORjr9v.png

This image shows different channels or feature maps or filters at a particular chosen layer l.Now, in order to capture the style of an image we would calculate how “correlated” these filters are to each other meaning how similar are these feature maps.But what is meant by correlation ?

Let’s understand it with the help of an example:

Let the first two channel in the above image be Red and Yellow.Suppose, the red channel captures some simple feature (say, vertical lines) and if these two channels were correlated then whenever in the image there is a vertical lines that is detected by Red channel then there will be a Yellow-ish effect of the second channel.

Now,let’s look at how to calculate these correlations (mathematically).

In-order to calculate a correlation between different filters or channels we calculate the dot-product between the vectors of the activations of the two filters.The matrix thus obtained is called Gram Matrix.

But how do we know whether they are correlated or not ?

If the dot-product across the activation of two filters is large then two channels are said to be correlated and if it is small then the images are un-correlated.Putting it mathematically :

Gram Matrix of Style Image(S):

Here k and k’ represents different filters or channels of the layer L. Let’s call this Gkk’[l][S].

https://miro.medium.com/v2/resize:fit:640/format:webp/0*L8Y_zB0tWkcxFKMh.png

Gram Matrix for style Image

Gram Matrix for Generated Image(G):

Here k and k’ represents different filters or channels of the layer L.Let’s call this Gkk’[l][G].

https://miro.medium.com/v2/resize:fit:640/format:webp/0*yjkYrNf7A_oMB_2V.png

Now,we are in the position to define Style loss:

Cost function between Style and Generated Image is the square of difference between the Gram Matrix of the style Image with the Gram Matrix of generated Image.

https://miro.medium.com/v2/resize:fit:720/format:webp/0*2LrpMFwbhD8OePdd.png

Total Loss Function :
The total loss function is the sum of the cost of the content and the style image.Mathematically,it can be expressed as :

https://miro.medium.com/v2/resize:fit:828/format:webp/0*JPXny-rYTIeZRSb4.png

You may have noticed Alpha and beta in the above equation.They are used for weighing Content and Style cost respectively.In general,they define the weightage of each cost in the Generated output image.

Once the loss is calculated,then this loss can be minimized using backpropagation which in turn will optimize our randomly generated image into a meaningful piece of art.

This sums up the working of Neural Style Transfer.

Implementation of Neural Style Transfer using TensorFlow :
Implementation is on my Github account :

https://github.com/blackburn07x/Convolutional-Neural-network/tree/main/Neural%20Style%20Transfer

Neural Style Transfer/Images/santa.jpg

Neural Style Transfer/Images/style1.jpg

Neural Style Transfer/Neural Style tranfer.py

Neural Style Transfer/__pycache__/VGGnst.cpython-37.pyc

Neural Style Transfer/nst_utils.py

Neural Style Transfer/output/0.png

Neural Style Transfer/output/120.png

Neural Style Transfer/output/140.png

Neural Style Transfer/output/160.png

Neural Style Transfer/output/160.png

Neural Style Transfer/output/180.png

Neural Style Transfer/output/20.png

Neural Style Transfer/output/200.png

Neural Style Transfer/output/220.png

Neural Style Transfer/output/240.png

Neural Style Transfer/output/40.png

Neural Style Transfer/output/60.png

Neural Style Transfer/output/80.png

Conculsion
In this blog, we went deep to see how Neural Style Transfer works.We also went through the mathematics behind NST. I would love to have a conversation in the comment section .Hope this contributes to your understanding of Neural Style Transfer
I would love to connect with you on #instagram .Thanks for sharing your time with me.

Neural Style Transfer with TensorFlow

This article will provide an overview of some of the core concepts underlying the technique. We will next go over neural style transfer in detail, as well as the basic conceptual grasp of this technique. We’ll look at the losses introduced by neural style transfer. Using this neural style transfer method, we will create a small project.

What is Neural Style Transfer?

Neural style transfer is an optimization technique used to take two images, a content image and a style reference image (such as an artwork by a famous painter), and blend them so the output image looks like the content image, but “painted” in the style of the style reference image. This technique is used by many popular Android iOS apps such as Prisma, DreamScope, and PicsArt.

https://media.geeksforgeeks.org/wp-content/uploads/20200820225906/styletransferexample.PNG

https://media.geeksforgeeks.org/wp-content/uploads/20200820225906/styletransferexample.PNG

(An example of style transfer A is a content image, B is output with style image in the bottom left corner)

VGG-19 Architecture overview

VGG-19 is a convolutional neural network (CNN) architecture from the VGG family of models. The Visual Graphics Group (VGG) at the University of Oxford introduced the VGG models, which are known for their simplicity and uniform architecture. VGG-19 has 19 layers, including 16 convolutional layers and 3 fully connected layers. The following are the key features of the VGG-19 architecture:

Input Layer:

Accepts 224×224 pixel images with three colour channels (RGB) as input.

Convolutional Blocks (Blocks 1–5):

VGG-19 is made up of five convolutional block sets. Each block is made up of several convolutional layers followed by max-pooling layers.

Convolutional layers commonly employ small 3×3 filters with a stride of one and rectified linear unit (ReLU) activation functions.

To reduce spatial dimensions, max-pooling layers with 2×2 filters and a stride of 2 are used.

Layers that are fully connected (FC6, FC7, and FC8):

There are three fully connected layers (FC6, FC7, and FC8) following the convolutional blocks.

The FC6 and FC7 layers each contain 4096 neurons and employ ReLU activation functions.

The FC8 layer (output layer) contains 1000 neurons with softmax activation, which correspond to the 1000 classes in the ImageNet dataset on which VGG-19 was trained.

Parameters:

Although VGG-19 is known for its simplicity, it has a large number of parameters, owing to the fully connected layers.

There are approximately 143.7 million trainable parameters in total.


Pre-trained Model:

VGG-19 is a popular pre-trained model for a variety of computer vision tasks. Researchers pre-trained it on large datasets such as ImageNet, allowing it to capture a diverse set of features from various categories.

The neural style transfer paper uses feature maps generated by intermediate layers of VGG-19 network to generate the output image. This architecture takes style and content images as input and stores the features extracted by convolution layers of VGG network. 

https://media.geeksforgeeks.org/wp-content/uploads/20200823113126/vgg19-architecture.png

Losses in Neural Style Transfer


Content Loss:

To calculate the content cost, we apply the mean square difference between matrices generated by the content layer, when we pass the generated image and the original image. Let p and x be the original image and the image that is generated, and P and F are their respective feature representation in layer l. We then define the squared-error loss between the two feature representations

L
content
(
ρ
,
x
,
L
)
=
1
2
∑
i
j
(
F
i
j
l
−
P
i
j
l
)
2
L 
content
​
 (ρ,x,L)= 
2
1
​
 ∑ 
ij
​
 (F 
ij
l
​
 −P 
ij
l
​
 ) 
2

Style Loss:

To calculate the style cost, we will first calculate the gram matrix.  The gram matrices calculation involves calculating the inner product between the vectorized feature maps of a particular layer. Here Gij (l) represents the inner product between vectorized features i,j of layer l.
G
i
j
l
=
∑
k
F
i
k
l
F
j
k
l
G 
ij
l
​
 =∑ 
k
​
 F 
ik
l
​
 F 
jk
l
​
 
Now to calculate the loss from a particular, we will find the mean square difference of gram matrices calculated from the feature vectors of the style image and the generated image. This then weighted to the layer weighing factor.

Let a and x be the original image and the generated image, and Al and Gl their respective style representation (gram matrices) in layer l. The contribution of layer l to the total loss is then:
E
l
=
1
4
N
l
2
M
l
2
∑
(
G
i
j
l
–
A
i
j
l
)
2
           
E 
l
​
 = 
4N 
l
2
​
 M 
l
2
​
 
1
​
 ∑(G 
ij
l
​
 –A 
ij
l
​
 ) 
2
  
Therefore, total style loss will be:
L
s
t
y
l
e
=
∑
l
=
0
L
w
l
E
l
L 
style
​
 =∑ 
l=0
L
​
 w 
l
​
 E 
l
​

Total Loss:

Total loss is the linear combination of style and content loss we defined above:
L
total
(
P
,
a
,
x
)
=
α
×
L
content
+
β
×
L
style
L 
total
​
 (P,a,x)=α×L 
content
​
 +β×L 
style
​
 

Where α and β are the weighting factors for content and style reconstruction, respectively.

Code Implementation in Tensorflow: 

First, we import the necessary module. In this post, we use TensorFlow v2 with Keras. We will also import VGG-19 model from tf.keras API.

Importing Libraries:

This script includes TensorFlow for deep learning, NumPy for numerical operations, Matplotlib for data visualisation, and Keras-specific components for working with pre-trained models and image processing.

# import numpy, tensorflow and matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# import VGG 19 model and keras Model API
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


Import Image data:

Now, we import the content and style images and save them into our working directory.

# Image Credits: Tensorflow Doc
content_path = tf.keras.utils.get_file(
'content.jpg',
'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file(
'style.jpg',
'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


Image processing:

Now, we load and process the image using Keras preprocess input in VGG 19. The expand_dims function adds a dimension to represent a number of images in the input. This preprocess_input function (used in VGG 19 ) converts the input RGB to BGR images and centre these values around 0 according to ImageNet data (no scaling).

# code to load and process image
def load_and_process_image(image_path):
	img = load_img(image_path)
	# convert image to array
	img = img_to_array(img)
	img = preprocess_input(img)
	img = np.expand_dims(img, axis=0)
	return img


Now, we define the deprocess function that takes the input image and perform the inverse of preprocess_input function that we imported above. To display the unprocessed image, we also define a display function.

# code
def deprocess(img):
	# perform the inverse of the pre processing step
	img[:, :, 0] += 103.939
	img[:, :, 1] += 116.779
	img[:, :, 2] += 123.68
	# convert RGB to BGR
	img = img[:, :, ::-1]

	img = np.clip(img, 0, 255).astype('uint8')
	return img


def display_image(image):
	# remove one dimension if image has 4 dimension
	if len(image.shape) == 4:
		img = np.squeeze(image, axis=0)

	img = deprocess(img)

	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img)
	return


Now, we use the above function to display the style and content images

# load content image
content_img = load_and_process_image(content_path)
display_image(content_img)

# load style image
style_img = load_and_process_image(style_path)
display_image(style_img)


Output:

https://media.geeksforgeeks.org/wp-content/uploads/20200820212333/content.png

(Content Image)

https://media.geeksforgeeks.org/wp-content/uploads/20200820212335/style.png

(Style Image)

Model Initialization:

Now, we initialize the VGG model with ImageNet weights, we will also remove the top layers and make it non-trainable.

# code
# this function download the VGG model and initialise it
model = VGG19(
	include_top=False,
	weights='imagenet'
)
# set training to False
model.trainable = False
# Print details of different layers

model.summary()


Output:

Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
80134624/80134624 [==============================] - 0s 0us/step
Model: "vgg19"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, None, None, 3)]   0         
                                                                 
 block1_conv1 (Conv2D)       (None, None, None, 64)    1792      
                                                                 
 block1_conv2 (Conv2D)       (None, None, None, 64)    36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, None, None, 64)    0         
                                                                 
 block2_conv1 (Conv2D)       (None, None, None, 128)   73856     
                                                                 
 block2_conv2 (Conv2D)       (None, None, None, 128)   147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, None, None, 128)   0         
                                                                 
 block3_conv1 (Conv2D)       (None, None, None, 256)   295168    
                                                                 
 block3_conv2 (Conv2D)       (None, None, None, 256)   590080    
                                                                 
 block3_conv3 (Conv2D)       (None, None, None, 256)   590080    
                                                                 
 block3_conv4 (Conv2D)       (None, None, None, 256)   590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, None, None, 256)   0         
                                                                 
 block4_conv1 (Conv2D)       (None, None, None, 512)   1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block4_conv4 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, None, None, 512)   0         
                                                                 
 block5_conv1 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block5_conv4 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, None, None, 512)   0         
                                                                 
=================================================================
Total params: 20024384 (76.39 MB)
Trainable params: 0 (0.00 Byte)
Non-trainable params: 20024384 (76.39 MB)
_________________________________________________________________

Content Model defining:

Now, we define the content and style model using Keras.Model API. The content model takes the image as input and output the feature map from “block5_conv1” from the above VGG model.

# define content model
content_layer = 'block5_conv2'
content_model = Model(
	inputs=model.input,
	outputs=model.get_layer(content_layer).output
)
content_model.summary()


Output:

Model: "functional_9"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, None, None, 3)]   0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, None, None, 64)    1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, None, None, 64)    36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, None, None, 64)    0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, None, None, 128)   73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, None, None, 128)   147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, None, None, 128)   0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, None, None, 256)   295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, None, None, 256)   590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, None, None, 256)   590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, None, None, 256)   590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, None, None, 256)   0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, None, None, 512)   1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, None, None, 512)   0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, None, None, 512)   2359808   
=================================================================
Total params: 15,304,768
Trainable params: 0
Non-trainable params: 15,304,768
_________________________________________________________________

Style Model defining:

Now, we define the content and style model using Keras.Model API. The style model takes an image as input and output the feature map from “block1_conv1, block3_conv1, and block5_conv2″ from the above VGG model.

# define style model
style_layers = [
	'block1_conv1',
	'block3_conv1',
	'block5_conv1'
]
style_models = [Model(inputs=model.input,
					outputs=model.get_layer(layer).output) for layer in style_layers]


Content Loss:

Now, we define the content loss function, it will take the feature map of generated and real images and calculate the mean square difference between them.

# Content loss
def content_loss(content, generated):
	a_C = content_model(content)
	a_G = content_model(generated) # Add this line to compute a_G
	loss = tf.reduce_mean(tf.square(a_C - a_G))
	return loss


Gram Matrix:

Now, we define the gram matrix function. This function also takes the real and generated images as the input of the model and calculates gram matrices of them before calculate the style loss weighted to different layers.

   

# gram matrix
def gram_matrix(A):
	channels = int(A.shape[-1])
	a = tf.reshape(A, [-1, channels])
	n = tf.shape(a)[0]
	gram = tf.matmul(a, a, transpose_a=True)
	return gram / tf.cast(n, tf.float32)


weight_of_layer = 1. / len(style_models)


Style Loss:

The function style_cost, defined by this code, determines the style loss between a generated image and a style image that is supplied. In neural style transfer algorithms, style loss is frequently employed to create an image that blends the content of two different images with their styles.

#style loss
def style_cost(style, generated):
	J_style = 0

	for style_model in style_models:
		a_S = style_model(style)
		a_G = style_model(generated)
		GS = gram_matrix(a_S)
		GG = gram_matrix(a_G)
		content_cost = tf.reduce_mean(tf.square(GS - GG))
		J_style += content_cost * weight_of_layer

	return J_style


Content Loss:

The content loss between a style image and a generated image is determined by the function content_cost, which is defined in this code. To make sure that the generated image preserves the original image’s content, neural style transfer algorithms frequently employ content loss.

#content loss
def content_cost(style, generated):
	J_content = 0

	for style_model in style_models:
		a_S = style_model(style)
		a_G = style_model(generated)
		GS = gram_matrix(a_S)
		GG = gram_matrix(a_G)
		content_cost = tf.reduce_mean(tf.square(GS - GG))
		J_content += content_cost * weight_of_layer

	return J_content


Training Function:

Now, we define our training function, we will train our model to 50 iterations. This model takes input images, the number of iterations as its argument.

# training function
generated_images = []


def training_loop(content_path, style_path, iterations=50, a=10, b=1000):
	# load content and style images from their respective path
	content = load_and_process_image(content_path)
	style = load_and_process_image(style_path)
	generated = tf.Variable(content, dtype=tf.float32)

	opt = tf.keras.optimizers.Adam(learning_rate=7)

	best_cost = math.inf
	best_image = None
	for i in range(iterations):
		start_time_cpu = time.process_time()
		start_time_wall = time.time()
		with tf.GradientTape() as tape:
			J_content = content_cost(style, generated)
			J_style = style_cost(style, generated)
			J_total = a * J_content + b * J_style

		grads = tape.gradient(J_total, generated)
		opt.apply_gradients([(grads, generated)])

		end_time_cpu = time.process_time() # Record end time for CPU
		end_time_wall = time.time() # Record end time for wall time
		cpu_time = end_time_cpu - start_time_cpu # Calculate CPU time
		wall_time = end_time_wall - start_time_wall # Calculate wall time

		if J_total < best_cost:
			best_cost = J_total
			best_image = generated.numpy()

		print("CPU times: user {} µs, sys: {} ns, total: {} µs".format(
		int(cpu_time * 1e6),
		int(( end_time_cpu - start_time_cpu) * 1e9),
		int((end_time_cpu - start_time_cpu + 1e-6) * 1e6))
			)
		
		print("Wall time: {:.2f} µs".format(wall_time * 1e6))
		print("Iteration :{}".format(i))
		print('Total Loss {:e}.'.format(J_total))
		generated_images.append(generated.numpy())

	return best_image


Model Training:

Now, we train our model using the training function we defined above.

# Train the model and get best image
final_img = training(content_path, style_path)


Output:

CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.2 µs
Iteration :0
Total Loss 5.133922e+11.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.72 µs
Iteration :1
Total Loss 3.510511e+11.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.68 µs
Iteration :2
Total Loss 2.069992e+11.
CPU times: user 3 µs, sys: 1e+03 ns, total: 4 µs
Wall time: 6.2 µs
Iteration :3
Total Loss 1.669609e+11.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.44 µs
Iteration :4
Total Loss 1.575840e+11.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.96 µs
Iteration :5
Total Loss 1.200623e+11.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.96 µs
Iteration :6
Total Loss 8.824594e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.72 µs
Iteration :7
Total Loss 7.168546e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.48 µs
Iteration :8
Total Loss 6.207320e+10.
CPU times: user 3 µs, sys: 1e+03 ns, total: 4 µs
Wall time: 8.34 µs
Iteration :9
Total Loss 5.390836e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.2 µs
Iteration :10
Total Loss 4.735992e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.96 µs
Iteration :11
Total Loss 4.301782e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.2 µs
Iteration :12
Total Loss 3.912694e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.68 µs
Iteration :13
Total Loss 3.445185e+10.
CPU times: user 0 ns, sys: 3 µs, total: 3 µs
Wall time: 6.2 µs
Iteration :14
Total Loss 2.975165e+10.
CPU times: user 2 µs, sys: 0 ns, total: 2 µs
Wall time: 5.96 µs
Iteration :15
Total Loss 2.590984e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 20 µs
Iteration :16
Total Loss 2.302116e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.72 µs
Iteration :17
Total Loss 2.082643e+10.
CPU times: user 4 µs, sys: 1e+03 ns, total: 5 µs
Wall time: 8.34 µs
Iteration :18
Total Loss 1.906701e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.25 µs
Iteration :19
Total Loss 1.759801e+10.
CPU times: user 3 µs, sys: 1e+03 ns, total: 4 µs
Wall time: 6.2 µs
Iteration :20
Total Loss 1.635128e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.2 µs
Iteration :21
Total Loss 1.525327e+10.
CPU times: user 3 µs, sys: 1e+03 ns, total: 4 µs
Wall time: 5.96 µs
Iteration :22
Total Loss 1.418364e+10.
CPU times: user 4 µs, sys: 1 µs, total: 5 µs
Wall time: 9.06 µs
Iteration :23
Total Loss 1.306596e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.25 µs
Iteration :24
Total Loss 1.196509e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.96 µs
Iteration :25
Total Loss 1.102290e+10.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.96 µs
Iteration :26
Total Loss 1.025539e+10.
CPU times: user 7 µs, sys: 3 µs, total: 10 µs
Wall time: 12.6 µs
Iteration :27
Total Loss 9.570500e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.72 µs
Iteration :28
Total Loss 8.917115e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.96 µs
Iteration :29
Total Loss 8.328761e+09.
CPU times: user 3 µs, sys: 1e+03 ns, total: 4 µs
Wall time: 9.54 µs
Iteration :30
Total Loss 7.840127e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.44 µs
Iteration :31
Total Loss 7.406647e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 8.34 µs
Iteration :32
Total Loss 6.967848e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.72 µs
Iteration :33
Total Loss 6.531650e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.72 µs
Iteration :34
Total Loss 6.136975e+09.
CPU times: user 2 µs, sys: 1 µs, total: 3 µs
Wall time: 5.96 µs
Iteration :35
Total Loss 5.788804e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.72 µs
Iteration :36
Total Loss 5.476942e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.2 µs
Iteration :37
Total Loss 5.204070e+09.
CPU times: user 3 µs, sys: 1 µs, total: 4 µs
Wall time: 6.2 µs
Iteration :38
Total Loss 4.954049e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.96 µs
Iteration :39
Total Loss 4.708641e+09.
CPU times: user 3 µs, sys: 2 µs, total: 5 µs
Wall time: 6.2 µs
Iteration :40
Total Loss 4.487677e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.96 µs
Iteration :41
Total Loss 4.296946e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.96 µs
Iteration :42
Total Loss 4.107909e+09.
CPU times: user 3 µs, sys: 1e+03 ns, total: 4 µs
Wall time: 6.44 µs
Iteration :43
Total Loss 3.918156e+09.
CPU times: user 3 µs, sys: 1e+03 ns, total: 4 µs
Wall time: 6.2 µs
Iteration :44
Total Loss 3.747263e+09.
CPU times: user 3 µs, sys: 1e+03 ns, total: 4 µs
Wall time: 8.34 µs
Iteration :45
Total Loss 3.595638e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 5.72 µs
Iteration :46
Total Loss 3.458928e+09.
CPU times: user 2 µs, sys: 1e+03 ns, total: 3 µs
Wall time: 6.2 µs
Iteration :47
Total Loss 3.331772e+09.
CPU times: user 4 µs, sys: 1e+03 ns, total: 5 µs
Wall time: 9.3 µs
Iteration :48
Total Loss 3.205911e+09.
CPU times: user 3 µs, sys: 1e+03 ns, total: 4 µs
Wall time: 5.96 µs
Iteration :49
Total Loss 3.089630e+09.

Model Prediction:

In the final step, we plot the final and intermediate results.

# code to display best generated image and last 10 intermediate results
plt.figure(figsize=(12, 12))

for i in range(10):
	plt.subplot(4, 3, i + 1)
	display_image(generated_images[i+39])
plt.show()

# plot best result
display_image(final_img)


Output:

https://media.geeksforgeeks.org/wp-content/uploads/20200820150016/l10res.PNG

(Last 10 generated images)

https://media.geeksforgeeks.org/wp-content/uploads/20200820224306/bestimage.png

(Best generated image)

https://www.geeksforgeeks.org/introduction-convolution-neural-network/

https://www.geeksforgeeks.org/activation-functions/

https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/

https://www.geeksforgeeks.org/vgg-16-cnn-model/

https://www.geeksforgeeks.org/tensorflow-2-0/

https://www.geeksforgeeks.org/python-tensorflow-tf-keras-layers-conv2d-function/

https://www.geeksforgeeks.org/python-tensorflow-tf-keras-layers-conv2d-function/

https://www.geeksforgeeks.org/python-introduction-matplotlib/









