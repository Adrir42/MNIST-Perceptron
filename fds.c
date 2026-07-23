#include "ocr.h"

/*Open the training files using the paths written in the ocr.h file*/
int    open_training_files(t_fds *fds)
{
    ssize_t n;
    ssize_t i = 0;
    ssize_t j = 0;
    ssize_t l = 0;
    unsigned char buf_header[8];

    fds->fd_labels = open(PATH_TRAINING_LABELS, O_RDONLY | O_BINARY);
    if (fds->fd_labels < 0)
        return (-1);
    fds->fd_images = open(PATH_TRAINING_IMAGES, O_RDONLY | O_BINARY);
    if (fds->fd_images < 0)
    {
        close(fds->fd_labels);
        return (-1);
    }

    //skip the header of the images files
    while (i < 2)
    {
        n = read(fds->fd_images, buf_header, sizeof(buf_header));
        j = 0;
        while (j < n)
        {
            fds->header_images[j + l] = buf_header[j];
            j++;
        }
        l = 8;
        i++;
    }

    //skip the header of the labels file
    n = read(fds->fd_labels, buf_header, sizeof(buf_header));
    j = 0;
    while (j < n)
    {
        fds->header_labels[j] = buf_header[j];
        j++;
    }
    return (0);
}

/*Open the test files, skipping the header and using the paths written in the ocr.h file*/
int    open_test_files(t_fds *fds)
{
    ssize_t n;
    ssize_t i = 0;
    unsigned char buf_header[8];

    fds->fd_test_labels = open(PATH_TEST_LABELS, O_RDONLY | O_BINARY);
    if (fds->fd_test_labels < 0)
        return (-1);
    fds->fd_test_images = open(PATH_TEST_IMAGES, O_RDONLY | O_BINARY);
    if (fds->fd_test_images < 0)
    {
        close(fds->fd_test_labels);
        return (-1);
    }

    //skip the header of the images files
    i = 0;
    while (i < 2)
    {
        n = read(fds->fd_test_images, buf_header, sizeof(buf_header));
        i++;
    }

    //skip the header of the labels file
    n = read(fds->fd_test_labels, buf_header, sizeof(buf_header));
    if (n != 8)
        return (-1);
    return (0);
}

void    close_train_fds(const t_fds *fds)
{
    close(fds->fd_labels);
    close(fds->fd_images);
}

void    close_test_fds(const t_fds *fds)
{
    close(fds->fd_test_labels);
    close(fds->fd_test_images);
}