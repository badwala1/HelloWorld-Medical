<script>
        async function loadModel() {
            // Load the trained model (e.g., in joblib or pickle format)
            const model = await tf.loadLayersModel('random_forest_model.json'); // Load your model here

            // Prepare input data as a tensor
            const formdata = new FormData(document.getElementById('input-form'));

            // Make predictions
            const predictions = model.predict(formdata);
            predictions.print(); // Print the predictions to the console
        }

        // Load the model when the page loads
        loadModel();
</script>
