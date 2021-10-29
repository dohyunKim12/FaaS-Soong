#!/usr/bin/env python
import pika

connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@faasoong.iptime.org:5672/'))
    #pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='accident')

channel.basic_publish(exchange='', routing_key='accident', body='37.495410&126.955841')
print(" [x] Sent 'accident occur!'")
connection.close()
