import pika, sys, os, time

def main():
    connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@116.89.189.12:5672/'))
    channel = connection.channel()

    channel.queue_declare(queue='hello')

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        b = time.time()
        print("duration : ", b-a)

    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)    
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        a = time.time()
        url = 'amqp://faasoong:tnd@faasoong.iptime.org:5672/'
        connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@116.89.189.12:5672/'))
            #pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='hello')
        channel.basic_publish(exchange='', routing_key='hello', body='100')
        print(" [x] Sent 'Hello World!'")
        connection.close()
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
