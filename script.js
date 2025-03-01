// Global variable to hold our model
let model;
let labels = [];

// Load the model when the page loads
window.addEventListener('DOMContentLoaded', async () => {
    try {
        // Show a loading message
        document.getElementById('prediction-result').textContent = 'Loading model...';
        
        // Try different model loading approaches
        try {
            // First try to load as a LayersModel (more common for converted models)
            model = await tf.loadLayersModel('model/model.json');
        } catch (layersError) {
            console.log('Could not load as layers model, trying graph model...');
            try {
                // If that fails, try as a GraphModel
                model = await tf.loadGraphModel('model/model.json');
            } catch (graphError) {
                // If both fail, throw a more detailed error
                throw new Error(`Failed to load model: ${layersError.message} and ${graphError.message}`);
            }
        }
        
        // Load metadata
        try {
            const metadataResponse = await fetch('model/metadata.json');
            const metadata = await metadataResponse.json();
            
            // Store the labels
            labels = metadata.labels || [];
            console.log("Model labels:", labels);
            
            // Initialize your UI based on metadata
            setupUI(metadata);
        } catch (metadataError) {
            console.log('Error loading metadata:', metadataError);
            // Setup default UI without metadata
            setupDefaultUI();
        }
        
        document.getElementById('prediction-result').textContent = 'Model loaded successfully. Ready for predictions.';
    } catch (error) {
        console.error('Error loading the model:', error);
        document.getElementById('prediction-result').textContent = 'Error loading the model: ' + error.message;
    }
});

// Setup default UI without metadata
function setupDefaultUI() {
    const inputSection = document.getElementById('input-section');
    
    // Assume it's an image model by default
    inputSection.innerHTML = `
        <input type="file" id="image-upload" accept="image/*">
        <div id="image-preview" style="margin-top: 10px; min-height: 200px; border: 1px dashed #ccc;"></div>
    `;
    
    // Add event listener for image upload
    document.getElementById('image-upload').addEventListener('change', previewImage);
}

// Setup the UI based on model metadata
function setupUI(metadata) {
    const inputSection = document.getElementById('input-section');
    
    // This is for an image model
    inputSection.innerHTML = `
        <h3>Upload Brain MRI Scan</h3>
        <input type="file" id="image-upload" accept="image/*">
        <div id="image-preview" style="margin-top: 10px; min-height: 200px; border: 1px dashed #ccc;"></div>
    `;
    
    // Add event listener for image upload
    document.getElementById('image-upload').addEventListener('change', previewImage);
    
    // Display available classifications
    const resultSection = document.getElementById('result-section');
    resultSection.innerHTML = `
        <h2>Results</h2>
        <div id="prediction-result"></div>
        <div id="classification-info">
            <h3>Possible Classifications:</h3>
            <ul>
                ${metadata.labels.map(label => `<li>${label}</li>`).join('')}
            </ul>
        </div>
    `;
}

// Example function to preview an uploaded image
function previewImage(event) {
    const preview = document.getElementById('image-preview');
    const file = event.target.files[0];
    const reader = new FileReader();
    
    reader.onload = function() {
        const img = document.createElement('img');
        img.src = reader.result;
        img.id = 'input-image';
        img.style.maxWidth = '100%';
        preview.innerHTML = '';
        preview.appendChild(img);
    }
    
    if (file) {
        reader.readAsDataURL(file);
    }
}

// Function to run prediction when the button is clicked
document.getElementById('predict-btn').addEventListener('click', async () => {
    if (!model) {
        alert('Model is not loaded yet. Please wait.');
        return;
    }
    
    try {
        // Example: If your model takes image input
        const img = document.getElementById('input-image');
        if (!img) {
            alert('Please upload an image first.');
            return;
        }
        
        // Preprocess the image (this will depend on your model's requirements)
        const tensor = preprocessImage(img);
        
        // Run inference
        const prediction = await model.predict(tensor);
        
        // Process and display results
        displayResults(prediction);
        
        // Clean up
        tensor.dispose();
    } catch (error) {
        console.error('Error during prediction:', error);
        document.getElementById('prediction-result').textContent = 'Error making prediction: ' + error.message;
    }
});

// Example function to preprocess an image for the model
function preprocessImage(img) {
    // This is just an example - adjust based on your model's requirements
    return tf.tidy(() => {
        // Convert image to tensor
        let tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([224, 224]) // Resize to model's expected size (from metadata)
            .toFloat();
            
        // Normalize (if required by your model)
        tensor = tensor.div(tf.scalar(255));
        
        // Add batch dimension
        return tensor.expandDims(0);
    });
}

// Function to display results with the specific class labels
function displayResults(prediction) {
    const resultElement = document.getElementById('prediction-result');
    
    // Convert tensor to array
    prediction.data().then(data => {
        // Create a formatted result display
        let resultHTML = '<div style="margin-top: 20px;">';
        resultHTML += '<h3>Analysis Results:</h3>';
        
        // If we have labels from metadata
        if (labels.length > 0 && labels.length === data.length) {
            // Find the highest probability class
            const maxIndex = data.indexOf(Math.max(...data));
            const maxProbability = data[maxIndex] * 100;
            
            // Add a clear result message
            resultHTML += `<p style="font-size: 18px; font-weight: bold; color: ${maxIndex === 0 ? 'red' : 'green'};">
                Prediction: ${labels[maxIndex]} (${maxProbability.toFixed(2)}%)
            </p>`;
            
            // Add a bar chart for all probabilities
            resultHTML += '<div style="margin-top: 15px;">';
            labels.forEach((label, i) => {
                const probability = data[i] * 100;
                resultHTML += `
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; align-items: center;">
                            <div style="width: 200px; overflow: hidden; text-overflow: ellipsis;">${label}:</div>
                            <div style="flex-grow: 1; margin-left: 10px;">
                                <div style="background-color: #e0e0e0; height: 20px; width: 100%; border-radius: 3px;">
                                    <div style="background-color: ${i === 0 ? '#ff6b6b' : '#4caf50'}; height: 100%; width: ${probability}%; border-radius: 3px;"></div>
                                </div>
                            </div>
                            <div style="width: 60px; text-align: right; margin-left: 10px;">${probability.toFixed(2)}%</div>
                        </div>
                    </div>
                `;
            });
            resultHTML += '</div>';
        } else {
            // If no labels are available, just show raw values
            resultHTML += `<p>Raw prediction values: ${Array.from(data).map(v => (v * 100).toFixed(2) + '%').join(', ')}</p>`;
            const maxIndex = data.indexOf(Math.max(...data));
            resultHTML += `<p>Highest probability class: ${maxIndex} (${(data[maxIndex] * 100).toFixed(2)}%)</p>`;
        }
        
        resultHTML += '</div>';
        resultElement.innerHTML = resultHTML;
    });
}