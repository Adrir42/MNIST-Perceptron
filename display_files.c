#include "ocr.h"

void    display_labels_file()
{
    int fd_labels = open("train-labels-idx1-ubyte/train-labels.idx1-ubyte", O_RDONLY | O_BINARY);
    unsigned char buf[8];
    int i = 0;
    ssize_t n;
    ssize_t j = 0;

    if (fd_labels == -1)
    {
        printf("Error while opening the label file\n");
        return ;
    }

    while ((n = read(fd_labels, buf, sizeof(buf))) > 0)
    {
        j = 0;
        while (j < n)
        {
            printf("%d", buf[j]);
            j++;
        }
        printf("\n");
        i += j;
    }
    printf("\n\n\n%d", i);
    close(fd_labels);
}

void    display_images_file()
{
    int fd_images = open("train-images-idx3-ubyte/train-images.idx3-ubyte", O_RDONLY | O_BINARY);
    unsigned char buf_header[8];
    unsigned char buf[28];
    int i = 0;
    ssize_t n;
    ssize_t j = 0;

    if (fd_images == -1)
    {
        printf("Error while opening the label file\n");
        return ;
    }

    printf("HEADER\n");
    while (i < 2 && (n = read(fd_images, buf_header, sizeof(buf_header))) > 0)
    {
        j = 0;
        while (j < n)
        {
            printf("%d", buf_header[j]);
            j++;
        }
        printf("\n");
        i++;
    }
    printf("IMAGES\n");
    i = 0;
    while ((n = read(fd_images, buf, sizeof(buf))) > 0)
    {
        j = 0;
        while (j < n)
        {
            printf("%d ", buf[j]);
            j++;
        }
        printf("\n");
        if (i % 28 == 0)
            printf("\n");
        i++;
    }
    close(fd_images);
}

void    display_test_labels_file()
{
    int fd_labels = open("t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte", O_RDONLY | O_BINARY);
    unsigned char buf[8];
    int i = 0;
    ssize_t n;
    ssize_t j = 0;

    if (fd_labels == -1)
    {
        printf("Error while opening the label file\n");
        return ;
    }

    while ((n = read(fd_labels, buf, sizeof(buf))) > 0)
    {
        j = 0;
        while (j < n)
        {
            printf("%d", buf[j]);
            j++;
        }
        printf("\n");
        i += j;
    }
    printf("\n\n\n%d", i);
    close(fd_labels);
}

void    display_test_images_file()
{
    int fd_images = open("t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", O_RDONLY | O_BINARY);
    unsigned char buf_header[8];
    unsigned char buf[28];
    int i = 0;
    ssize_t n;
    ssize_t j = 0;

    if (fd_images == -1)
    {
        printf("Error while opening the label file\n");
        return ;
    }

    printf("HEADER\n");
    while (i < 2 && (n = read(fd_images, buf_header, sizeof(buf_header))) > 0)
    {
        j = 0;
        while (j < n)
        {
            printf("%d", buf_header[j]);
            j++;
        }
        printf("\n");
        i++;
    }
    printf("IMAGES\n");
    i = 0;
    while ((n = read(fd_images, buf, sizeof(buf))) > 0)
    {
        j = 0;
        while (j < n)
        {
            printf("%d ", buf[j]);
            j++;
        }
        printf("\n");
        if (i % 28 == 0)
            printf("\n");
        i++;
    }
    close(fd_images);
}