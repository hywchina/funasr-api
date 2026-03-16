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
  --restart=always \
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


# 赤峰项目更新说明 

## 整体替换逻辑
1. kotaemon
step0: 把项目下载到硬盘

step1:  两个文件替换
a. /Users/ai_diagnosis/projects/kotaemon/libs/ktem/ktem/pages/voice_assistant.py 新增
b. /Users/ai_diagnosis/projects/kotaemon/libs/ktem/ktem/main.py 修改


2. funasr-api 
step0: 把项目下载到硬盘，把镜像下载到硬盘

step1: 加载镜像 
docker load -i funasr-api:v1.0.0.tar 

step2: 启动容器
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
  --restart=always \
  funasr-api:v1.0.0

  <!-- --restart=always
  --restart unless-stopped  -->
  
测试 ：
http://localhost:17003/ws/v1/asr/test ： 正常
http://10.42.0.219:17003/ws/v1/asr/test ：麦克风启动失败: Cannot read properties of undefined (reading 'getUserMedia')

https://10.42.0.219:17003/ws/v1/asr/test：黑屏

step3: 添加反向代理
bash scripts/fix_container_https.sh

step4: 测试
1. http://localhost:17003/ws/v1/asr/test   ： 400 Bad Request
2. http://10.42.0.219:17003/ws/v1/asr/test： 400 Bad Request
3. https://10.42.0.219:17003/ws/v1/asr/test ： 正常 
