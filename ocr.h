#ifndef OCR_H
# define OCR_H

/*
INCLUDES
*/

# include <stdlib.h>
# include <fcntl.h>
# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <time.h>

/*
CONSTANTS
*/

//Input (28*28=784 pixels) → [Hidden layer: linear + ReLU] → (128) 
//→ [Output layer : linear + softmax] → (10)
# define LEARNING_RATE 0.05
# define NB_EPOCHS 20
# define NB_IMAGES_TRAINING_SET 60000
# define NB_IMAGES_TEST_SET 10000
# define IMAGE_SIZE 784
# define NB_HIDDEN_NEURONS 128
# define NB_OUTPUT_NEURONS 10

//Paths to the dataset
# define PATH_TRAINING_LABELS "../train-labels-idx1-ubyte/train-labels.idx1-ubyte"
# define PATH_TRAINING_IMAGES "../train-images-idx3-ubyte/train-images.idx3-ubyte"
# define PATH_TEST_LABELS "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
# define PATH_TEST_IMAGES "../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"

/*
STRUCTURES
*/

typedef struct s_image
{
	float	        data[IMAGE_SIZE];
    int             label;
}					t_image;

typedef struct s_layer_hidden
{
    float   weights[IMAGE_SIZE][NB_HIDDEN_NEURONS];
    float   biases[NB_HIDDEN_NEURONS];

    float   weights_grad[IMAGE_SIZE][NB_HIDDEN_NEURONS];   // gradients of weights, same size as weights
    float   biases_grad[NB_HIDDEN_NEURONS];                // gradients of biases, same size as biases
}   t_hidden;

typedef struct s_layer_output
{
    float   weights[NB_HIDDEN_NEURONS][NB_OUTPUT_NEURONS];
    float   biases[NB_OUTPUT_NEURONS];

    float   weights_grad[NB_HIDDEN_NEURONS][NB_OUTPUT_NEURONS];
    float   biases_grad[NB_OUTPUT_NEURONS];
}   t_output;

typedef struct s_network
{
    t_hidden hidden;
    t_output output;
}   t_network;

typedef struct s_fds
{
	int             fd_images;
    int             fd_labels;
    int             fd_test_images;
    int             fd_test_labels;
    unsigned char   header_images[16];
    unsigned char   header_labels[8];
}					t_fds;

typedef struct s_sums
{
    float   z1[NB_HIDDEN_NEURONS];
    float   a1[NB_HIDDEN_NEURONS];
    float   z2[NB_OUTPUT_NEURONS];
    float   a2[NB_OUTPUT_NEURONS];
    int     prediction_result;
    float   loss;
    float   prob_max;
}					t_sums;

typedef struct s_grads
{
    float   z1[NB_HIDDEN_NEURONS];
    float   a1[NB_HIDDEN_NEURONS];
    float   z2[NB_OUTPUT_NEURONS];
}					t_grads;

/*
FUNCTIONS
*/

/*
//display_files.c
void    display_labels_file();
void    display_images_file();
void    display_test_labels_file();
void    display_test_images_file();

//display.c
void    display_image(t_image *image);
void    display_hidden(t_hidden *layer);
void    display_output(t_output *layer);
void    display_sums(t_sums *sums);
*/

//fds.c
int     open_training_files(t_fds *fds);
int     open_test_files(t_fds *fds);
void    close_train_fds(const t_fds *fds);
void    close_test_fds(const t_fds *fds);

//init.c
float   get_random_weight_He();
float   get_random_weight_Xavier_Glorot();
void init_hidden_layer(t_hidden *hidden);
void init_output_layer(t_output *output);
void    init_sums(t_sums *sums);

//forward_pass.c
void    forward_pass_hidden(t_sums *sum, const t_image *image, const t_network *network);
void    forward_pass_output(t_sums *sum, const t_network *network);
void    softmax(t_sums *sum);
void    make_prediction(t_sums *sums);
void    cross_entropy_loss(t_sums *sums, const t_image *image);
void    forward_pass(t_sums *sums, t_image *image, t_network *network);

//backward_pass.c
void    init_grads(t_grads *grads, t_network *network);
void    backward_pass_output(const t_sums *sums, t_grads *grads, const t_image *image, t_output *output);
void    backward_pass_hidden(const t_sums *sums, t_grads *grads, const t_image *image, t_network *network);
void    backward_pass(t_sums *sums, t_grads *grads, t_image *image, t_network *network);
void    update_weights_and_biases(t_network *network);

#endif