document.addEventListener('DOMContentLoaded', () => {
    // Particle Background
    particlesJS('particles-js', {
        particles: {
            number: { value: 80, density: { enable: true, value_area: 800 } },
            color: { value: '#00d4ff' },
            shape: { type: 'circle' },
            opacity: { value: 0.5, random: true },
            size: { value: 3, random: true },
            line_linked: { enable: true, distance: 150, color: '#00d4ff', opacity: 0.4, width: 1 },
            move: { enable: true, speed: 2, direction: 'none', random: false, straight: false, out_mode: 'out' }
        },
        interactivity: {
            detect_on: 'canvas',
            events: { onhover: { enable: true, mode: 'repulse' }, onclick: { enable: true, mode: 'push' }, resize: true },
            modes: { repulse: { distance: 100, duration: 0.4 }, push: { particles_nb: 4 } }
        },
        retina_detect: true
    });

    // Index page functionality
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const fileMessage = document.querySelector('.file-message');

    if (dropArea && fileInput) {
        dropArea.addEventListener('click', () => fileInput.click());

        dropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropArea.classList.add('dragover');
        });

        dropArea.addEventListener('dragleave', () => {
            dropArea.classList.remove('dragover');
        });

        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dropArea.classList.remove('dragover');
            fileInput.files = e.dataTransfer.files;
            updateFileMessage();
        });

        fileInput.addEventListener('change', updateFileMessage);

        function updateFileMessage() {
            if (fileInput.files.length > 0) {
                fileMessage.textContent = `Selected: ${fileInput.files[0].name}`;
            }
        }

        const form = document.getElementById('uploadForm');
        form.addEventListener('submit', () => {
            const btn = form.querySelector('.analyze-btn');
            btn.textContent = 'Analyzing...';
            btn.disabled = true;
        });
    }

    // Result page functionality
    const resultBox = document.querySelector('.result-box');
    if (resultBox) {
        setTimeout(() => {
            resultBox.style.opacity = '1';
        }, 100);
    }
});