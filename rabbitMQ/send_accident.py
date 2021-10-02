#!/usr/bin/env python
import pika

connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@116.89.189.12:5672/'))
    #pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='accident')

channel.basic_publish(exchange='', routing_key='accident', body='36.2325&127.38321')
print(" [x] Sent 'motion detect!'")
connection.close()
