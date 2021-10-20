#!/usr/bin/env python
import pika, sys, os

def main():
    connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@faasoong.iptime.org:5672/'))
    channel = connection.channel()

    channel.queue_declare(queue='motion')
#    for method_frame, properties, body in channel.consume('motion'):
          # Display the message parts and acknowledge the message
#          print(method_frame, properties, body)
#          channel.basic_ack(method_frame.delivery_tag)

          # Escape out of the loop after 10 messages
#          if method_frame.delivery_tag == 1:
#              break

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)

    channel.basic_consume(queue='motion', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

    channel.k

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
