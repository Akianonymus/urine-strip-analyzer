from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        file_name = default_storage.save(image_file.name, ContentFile(image_file.read()))
        file_path = default_storage.path(file_name)

        # Process the image
        colors = process_image(file_path)

        # Delete the file after processing
        default_storage.delete(file_name)

        return JsonResponse({'colors': colors})

    return JsonResponse({'error': 'Invalid request'}, status=400)

def process_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    num_pixels = image_array.shape[0] * image_array.shape[1]
    image_array_reshaped = image_array.reshape(num_pixels, -1)
    num_colors = 10
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image_array_reshaped)
    colors = kmeans.cluster_centers_
    colors = colors.astype(int)
    result = []
    for color in colors:
        result.append(color.tolist())
    return result

def index(request):
    return render(request, 'index.html')

