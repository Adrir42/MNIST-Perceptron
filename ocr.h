#ifndef OCR_H
# define OCR_H

# include <stdlib.h>
# include <fcntl.h>
# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <time.h>

/*
Input (28*28=784 pixels) → [Hidden layer: linear + ReLU] → (128) → [Output layer : linear + softmax] → (10)
*/

# define LEARNING_RATE 0.05

# define PATH_TRAINING_LABELS "../train-labels-idx1-ubyte/train-labels.idx1-ubyte"
# define PATH_TRAINING_IMAGES "../train-images-idx3-ubyte/train-images.idx3-ubyte"
# define PATH_TEST_LABELS "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
# define PATH_TEST_IMAGES "../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"

typedef struct s_image
{
	float	        data[784];
    int             label;
}					t_image;

typedef struct s_layer_hidden
{
    int     nb_inputs;      // (784 or 128)
    int     nb_outputs;     // (128 or 10)

    float   weights[784][128];       // size = nb_inputs * nb_outputs
    float   biases[128];        // size = nb_outputs

    float   weights_grad[784][128];  // gradients of weights, same size as weights
    float   biases_grad[128];   // gradients of biases, same size as biases
}   t_hidden;

typedef struct s_layer_output
{
    int     nb_inputs;
    int     nb_outputs;

    float   weights[128][10];
    float   biases[10];

    float   weights_grad[128][10];
    float   biases_grad[10];
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
    float   z1[128];
    float   a1[128];
    float   z2[10];
    float   a2[10];
    int     prediction_result;
    float   loss;
    float   prob_max;
}					t_sums;

typedef struct s_grads
{
    float   z1[128];
    float   a1[128];
    float   z2[10];
}					t_grads;

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
float   get_random_weight_He(t_hidden *hidden);
float   get_random_weight_Xavier_Glorot(t_output *output);
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