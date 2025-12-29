# ☁️ Syntropy Quant: Cloud Deployment Guide

To prevent your trading system from stopping when your laptop sleeps, you should deploy it to a **Cloud VPS (Virtual Private Server)**.

## 1. Recommended Cloud Providers
- **AWS (EC2)**: Use the `t3.micro` instance (Free Tier available).
- **DigitalOcean**: Use a $6/month "Droplet" (Ubuntu).
- **GCP (Compute Engine)**: Use the `e2-micro` instance (Always Free tier).

## 2. Server Setup (Ubuntu 22.04+ recommended)
Once you have your server, SSH into it and run:

```bash
# Update and install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

## 3. Deployment Steps
On your local machine, ensure your code is pushed to a private GitHub repository. Then on the server:

```bash
# 1. Clone your private repo
git clone https://github.com/YOUR_USERNAME/syntropy-quant.git
cd syntropy-quant

# 2. Create your .env file
# Copy your ALPACA_API_KEY, SECRET_KEY, etc. into this file
nano .env 

# 3. Start the system in the background using Docker
docker-compose up -d --build
```

## 4. Monitoring
- **View Logs**: `docker logs -f syntropy-quant-live`
- **Check Status**: `docker ps`
- **Stop System**: `docker-compose down`

## 5. Why Docker?
- **Isolation**: It won't mess with your server's Python version.
- **Persistence**: Your models and logs are saved in the server's folders.
- **Auto-Restart**: If the server reboots, the trading system starts automatically.

---
**Syntropy Quant v5.0** - *Symmetry is the key to robustness.*
