## funasr-api 修改的代码



docker-compose -f docker-compose-cpu.yml up -d      

docker-compose -f docker-compose-cpu.yml down



## 构建镜像
docker build -f Dockerfile.cpu -t funasr-api:v1.0.0 .


## 构建容器
docker run -d \
  --name funasr-api-cpu \
  -p 17003:8000 \
  -v "$(pwd)/temp:/app/temp" \
  -v "$(pwd)/logs:/app/logs" \
  -e API_KEY="" \
  -e ENABLED_MODELS="paraformer-large" \
  -e MODELSCOPE_PATH="/root/.cache/modelscope/hub/models" \
  -e HF_HUB_OFFLINE="" \
  -e HF_ENDPOINT="" \
  -e NGINX_RATE_LIMIT_RPS="0" \
  -e NGINX_RATE_LIMIT_BURST="0" \
  --restart unless-stopped \
  funasr-api:v1.0.0

## 删除容器和镜像
docker rm -f funasr-api-cpu && docker rmi -f funasr-api:v1.0.0

## 打包镜像 && 加载镜像 
docker save -o funasr-api:v1.0.0.tar  funasr-api:v1.0.0



# 进入容器执行
## 修改时间
echo "Asia/Shanghai" > /etc/timezone
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

## 安装vim
apt-get update && apt-get install -y vim

## 安装settools
uv pip install setuptools==70.0.0
