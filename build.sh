# If you get a permission denied running this, you need to add yourself to the docker group: sudo usermod -aG docker $USER
# Nuance: Rather than calling docker build ., which collects build context into the image, I use 
# docker build - < Dockerfile to pipe the Dockerfile into stdin, resulting in no build context.
# I am running the code remotely through a PyCharm connection to the container it creates.
docker build -t dads_replication:v1 - < Dockerfile
