#!/usr/bin/env python
import pika, sys, os

def main():
    connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@faasoong.iptime.org:5672/'))
    channel = connection.channel()

    channel.queue_delete(queue='motion')

    channel.queue_delete(queue='accident')
    channel.queue_delete(queue='fire')
    channel.queue_delete(queue='gun')
    channel.queue_delete(queue='knife')



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
