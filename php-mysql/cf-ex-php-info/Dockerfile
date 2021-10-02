FROM ryunchang/php-mysql:4.2

ENV MYSQL_ROOT_PASSWORD=tnd


EXPOSE 80 3306 33060

RUN chmod +x /init.sh

CMD ["mysqld"]

RUN ./init.sh

