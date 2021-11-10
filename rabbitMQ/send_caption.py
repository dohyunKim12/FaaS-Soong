#!/usr/bin/env python
import pika

connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@faasoong.iptime.org:5672/'))
    #pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='caption')

channel.basic_publish(exchange='', routing_key='caption', body='2021-11-09-20:30:4637.494779126.95916.png')
print(" [x] Sent 'accident occur!'")
connection.close()
