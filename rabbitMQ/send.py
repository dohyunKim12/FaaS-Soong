#!/usr/bin/env python
import pika

url = 'amqp://faasoong:tnd@faasoong.iptime.org:5672/'
connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@116.89.189.12:5672/'))
    #pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')

channel.basic_publish(exchange='', routing_key='hello', body='100')
print(" [x] Sent 'Hello World!'")
connection.close()
