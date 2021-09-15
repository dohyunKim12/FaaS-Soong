#!/usr/bin/env python
import pika

connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@116.89.189.12:5672/'))
    #pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='motion')

channel.basic_publish(exchange='', routing_key='motion', body='100')
print(" [x] Sent 'motion detect!'")
connection.close()
