#!/bin/bash
# ================================================================
# Build & Push TotalSpineSeg Docker container
# ================================================================
DOCKER_USERNAME="go2432"
IMAGE_NAME="totalspineseg"
TAG="latest"

echo "=============================================="
echo " Building TotalSpineSeg container..."
echo "=============================================="

docker build -f Dockerfile.totalspineseg -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} .

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build FAILED. Check errors above."
    exit 1
fi

echo ""
echo "✅ Build complete!"
echo ""
echo "Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo ""
echo "----------------------------------------------"
echo "To push to Docker Hub:"
echo "  docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo ""
echo "To run interactively with GPU:"
echo "  docker run -it --gpus all \\"
echo "    -v /path/to/input:/app/data \\"
echo "    -v /path/to/output:/app/output \\"
echo "    ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo ""
echo "To run inference directly:"
echo "  docker run --rm --gpus all \\"
echo "    -v /path/to/input:/app/data \\"
echo "    -v /path/to/output:/app/output \\"
echo "    ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} \\"
echo "    totalspineseg /app/data /app/output --iso"
echo "----------------------------------------------"
