document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const audioUpload = document.getElementById('audio-upload');
    const textInput = document.getElementById('text-input');
    const predictButton = document.getElementById('predict-button');
    const audioPlayer = document.getElementById('audio-player');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const resultsSection = document.getElementById('results');
    const emotionBadge = document.getElementById('emotion-badge');
    const confidenceSpan = document.getElementById('confidence');
    const confidenceBar = document.getElementById('confidence-bar');
    const modelComparison = document.getElementById('model-comparison');
    let chart = null;

    // Emotion color map
    const emotionColors = {
        happy: '#4CAF50',
        sad: '#2196F3',
        angry: '#F44336',
        disgust: '#9C27B0',
        fear: '#FF9800',
        surprise: '#FFEB3B',
        neutral: '#9E9E9E'
    };

    // Audio file handling with transcription
    audioUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (file) {
            // Set audio player source
            const audioURL = URL.createObjectURL(file);
            audioPlayer.src = audioURL;
            audioPlayerContainer.style.display = 'block';
            audioPlayer.load();

            // Transcribe audio automatically
            const formData = new FormData();
            formData.append('audio', file);
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) {
                    throw new Error('Transcription failed');
                }
                const data = await response.json();
                textInput.value = data.text; // Pre-fill text area with transcription
            } catch (error) {
                console.error('Transcription error:', error);
                alert('Failed to transcribe audio. Please enter text manually.');
                textInput.value = ''; // Clear text area on failure
            }
        }
    });

    // Drag and drop for audio files
    const uploadArea = document.querySelector('.file-upload-area');
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length && e.dataTransfer.files[0].type.includes('audio')) {
            audioUpload.files = e.dataTransfer.files;
            const event = new Event('change');
            audioUpload.dispatchEvent(event); // Trigger change event to handle transcription
        } else {
            alert('Please upload an audio file (WAV format)');
        }
    });

    // Prediction handler
    predictButton.addEventListener('click', async () => {
        const file = audioUpload.files[0];
        const text = textInput.value.trim();
        
        if (!file && !text) {
            alert('Please upload an audio file or enter text');
            return;
        }

        const formData = new FormData();
        if (file) formData.append('audio', file);
        if (text) formData.append('text', text);

        try {
            // Show loading state
            predictButton.disabled = true;
            predictButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Prediction failed');
            }

            const data = await response.json();
            displayResults(data);
            
        } catch (error) {
            console.error('Prediction error:', error);
            alert(`Error: ${error.message}`);
        } finally {
            predictButton.disabled = false;
            predictButton.innerHTML = '<i class="fas fa-magic"></i> Analyze Emotions';
        }
    });

    // Display results
    function displayResults(data) {
        resultsSection.style.display = 'block';
        const ensembleResult = data.results.Ensemble;
        
        // Set emotion badge
        emotionBadge.textContent = ensembleResult.emotion;
        emotionBadge.style.backgroundColor = emotionColors[ensembleResult.emotion.toLowerCase()] || '#88a47c';
        
        // Set confidence
        const confidencePercent = (ensembleResult.confidence * 100).toFixed(1);
        confidenceSpan.textContent = `${confidencePercent}%`;
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceBar.style.backgroundColor = getConfidenceColor(ensembleResult.confidence);
        
        // Create chart
        updateChart(data.emotion_classes, ensembleResult.probs);
        
        // Show model comparison
        showModelComparison(data.results);
    }

    // Update chart
    function updateChart(labels, probs) {
        const ctx = document.getElementById('emotion-chart').getContext('2d');
        
        if (chart) {
            chart.destroy();
        }
        
        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability',
                    data: probs,
                    backgroundColor: labels.map(label => emotionColors[label.toLowerCase()] || '#88a47c'),
                    borderColor: '#3a4a3a',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Probability',
                            color: '#3a4a3a'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Emotions',
                            color: '#3a4a3a'
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.raw.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Show model comparison
    function showModelComparison(results) {
        let comparisonHTML = '';
        
        for (const [modelName, result] of Object.entries(results)) {
            if (modelName === 'Ensemble') continue;
            
            comparisonHTML += `
                <div class="model-result">
                    <div class="model-name">${modelName}</div>
                    <div class="model-emotion" style="color: ${emotionColors[result.emotion.toLowerCase()] || '#3a4a3a'}">
                        ${result.emotion} (${(result.confidence * 100).toFixed(1)}%)
                    </div>
                </div>
            `;
        }
        
        modelComparison.innerHTML = comparisonHTML;
    }

    // Helper function for confidence color
    function getConfidenceColor(confidence) {
        const hue = confidence * 120; // 0 (red) to 120 (green)
        return `hsl(${hue}, 80%, 50%)`;
    }
});