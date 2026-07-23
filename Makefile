CC      = gcc
CFLAGS  = -Wall -Wextra -Werror
NAME    = perceptron

SRCS    = main.c fds.c init.c forward_pass.c backward_pass.c
OBJS    = $(SRCS:.c=.o)

all: $(NAME)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(NAME): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(NAME)

clean:
	rm -f $(OBJS)

fclean: clean
	rm -f $(NAME)

re: fclean all