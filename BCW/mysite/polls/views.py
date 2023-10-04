import tempfile
from io import BytesIO

from django.http import HttpResponse
from django.shortcuts import render
import joblib
import reportlab
from xhtml2pdf import pisa
from django.http import HttpResponse
from django.template.loader import get_template
from django.conf import settings
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from xhtml2pdf import pisa


def index(request):
    if request.method == 'POST':
        # Get the form input values
        patient=float(request.POST.get('patient'))
        radius_mean = float(request.POST.get('radius_mean'))
        texture_mean = float(request.POST.get('texture_mean'))
        smoothness_mean = float(request.POST.get('smoothness_mean'))
        compactness_mean = float(request.POST.get('compactness_mean'))
        symmetry_mean = float(request.POST.get('symmetry_mean'))
        fractal_dimension_mean = float(request.POST.get('fractal_dimension_mean'))
        radius_se = float(request.POST.get('radius_se'))
        texture_se = float(request.POST.get('texture_se'))
        smoothness_se = float(request.POST.get('smoothness_se'))
        compactness_se = float(request.POST.get('compactness_se'))
        symmetry_se = float(request.POST.get('symmetry_se'))
        fractal_dimension_se = float(request.POST.get('fractal_dimension_se'))

        # Load the pre-trained classification model
        model = joblib.load('C:/Users/pc/Desktop/Projet Integré S4/BCW/knn_model.joblib')

        # Perform the cancer classification
        features = [
            radius_mean, texture_mean, smoothness_mean, compactness_mean, symmetry_mean, fractal_dimension_mean,
            radius_se, texture_se, smoothness_se, compactness_se, symmetry_se, fractal_dimension_se
        ]
        prediction = model.predict([features])
        if prediction == 1:
            prediction_label = "malignant"
        else:
            prediction_label = "benign"

        # Render the result template with the classification prediction
        return render(request, 'result.html', {
            'patient':patient,
            'prediction': prediction_label,
            'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'smoothness_mean': smoothness_mean,
            'compactness_mean': compactness_mean,
            'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean,
            'radius_se': radius_se,
            'texture_se': texture_se,
            'smoothness_se': smoothness_se,
            'compactness_se': compactness_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
        })

        # Render the result template with the classification predictio

    # Render the form template for GET requests
    return render(request, 'form.html')

def generate_pdf(request):
    if request.method == 'POST':
        radius_mean = request.POST.get('radius_mean')
        texture_mean = request.POST.get('texture_mean')
        smoothness_mean = request.POST.get('smoothness_mean')
        compactness_mean = request.POST.get('compactness_mean')
        symmetry_mean = request.POST.get('symmetry_mean')
        fractal_dimension_mean =request.POST.get('fractal_dimension_mean')
        radius_se = request.POST.get('radius_se')
        texture_se = request.POST.get('texture_se')
        smoothness_se = request.POST.get('smoothness_se')
        compactness_se = request.POST.get('compactness_se')
        symmetry_se = request.POST.get('symmetry_se')
        fractal_dimension_se = request.POST.get('fractal_dimension_se')
        patient = request.POST.get('patient')

    template = get_template('result.html')
    context = {
        'patient':patient,
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'prediction': 'Malignant'
    }

    html = template.render(context)

    # Create a temporary file to store the PDF
    output_filename = 'C:/Users/pc/Downloads/report.pdf'

    # Generate the PDF using reportlab and pisa
    pdf = BytesIO()
    pisa.CreatePDF(BytesIO(html.encode("UTF-8")), pdf)

    # Save the PDF to a file
    with open(output_filename, 'wb') as pdf_file:
        pdf_file.write(pdf.getvalue())

    # Read the PDF file and return as response
    with open(output_filename, 'rb') as pdf_file:
        response = HttpResponse(pdf_file.read(), content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="C:/Users/pc/Downloads/report.pdf"'
        return response

    return HttpResponse("Error generating PDF", status=500)