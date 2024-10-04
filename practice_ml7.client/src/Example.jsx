import { useEffect, useState } from 'react';

const Example = () => {
    const [data, setData] = useState([]);
    const [selectedProduct, setSelectedProduct] = useState('');
    const [predictionValue, setPredictionValue] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const apiUrls = [
                    'http://localhost:5000/api/Products',
                    'https://localhost:5000/api/Products'
                ];

                let response;

                for (const url of apiUrls) {
                    console.log(`Attempting to fetch from: ${url}`);
                    try {
                        response = await fetch(url, {
                            method: 'GET',
                            mode: 'cors',
                        });
                        if (response.ok) break;
                    } catch (error) {
                        console.error(`Error fetching from ${url}:`, error);
                    }
                }

                if (!response || !response.ok) {
                    throw new Error(`HTTP error! status: ${response ? response.status : 'No response'}`);
                }

                const fetchedData = await response.json();
                console.log('Data:', fetchedData);
                setData(fetchedData);
                console.log('Data fetched successfully. Check the console for results.');

            } catch (error) {
                console.error('Error fetching data:', error);
                console.log('Data fetched unsuccessfully. Check the console for results.');
            }
        };

        fetchData();
    }, []);

    const handleProductChange = async (event) => {
        const selectedId = event.target.value;
        setSelectedProduct(selectedId);

        if (selectedId) {
            await callMLActionPredictionAPI();
            const selectedProductData = data.find(p => p.idProduct === parseInt(selectedId));
            if (selectedProductData) {
                await callMLActionAPI(selectedId, selectedProductData.name);
            }
        }
    };

    const callMLActionPredictionAPI = async () => {
        const apiUrls = [
            'http://localhost:5000/api/Products/MLActionPrediction',
            'https://localhost:5000/api/Products/MLActionPrediction'
        ];

        let response;

        try {
            for (const url of apiUrls) {
                console.log(`Attempting to call MLActionPrediction API: ${url}`);
                try {
                    response = await fetch(url, {
                        method: 'GET',
                        mode: 'cors',
                    });
                    if (response.ok) break;
                } catch (error) {
                    console.error(`Error calling MLActionPrediction API from ${url}:`, error);
                }
            }

            if (!response || !response.ok) {
                throw new Error(`HTTP error! status: ${response ? response.status : 'No response'}`);
            }

            const result = await response.json();
            console.log('MLActionPrediction API response:', result);

            // Extract the predictionDataUpdate value from the response
            setPredictionValue(result.predictionDataUpdate);
        } catch (error) {
            console.error('Error calling MLActionPrediction API:', error);
            setPredictionValue(null);
        }
    };

    // Step 15: Define function to call MLAction API
    // This function is called when a product is selected to perform some machine learning action
    const callMLActionAPI = async (id, name) => {
        // Define both HTTP and HTTPS URLs for the MLAction API
        const apiUrls = [
            `http://localhost:5000/api/Products/MLAction?id=${id}&name=${encodeURIComponent(name)}`,
            `https://localhost:5000/api/Products/MLAction?id=${id}&name=${encodeURIComponent(name)}`
        ];
        // Variable to store the API response
        let response;
        try {
            // Attempt to call the API using each URL until successful
            for (const url of apiUrls) {
                console.log(`Attempting to call MLAction API: ${url}`);
                try {
                    response = await fetch(url, {
                        method: 'GET', // Use GET method for this API call
                        mode: 'cors', // Enable Cross-Origin Resource Sharing
                    });
                    // If the call is successful, break the loop
                    if (response.ok) break;
                } catch (error) {
                    console.error(`Error calling MLAction API from ${url}:`, error);
                }
            }
            // Check if a successful response was received
            if (!response || !response.ok) {
                throw new Error(`HTTP error! status: ${response ? response.status : 'No response'}`);
            }
            // Parse and log the API response
            const result = await response.json();
            console.log('MLAction API response:', result);
        } catch (error) {
            console.error('Error calling MLAction API:', error);
        }
    };

    return (
        <div className="container">
            {predictionValue !== null && (
                <div className="prediction-data">
                    <h3>Prediction Data Update</h3>
                    <p>The prediction after {"\n"} 1-Use the selected price as a constant{"\n"}2-Train the model based on the selection{"\n"}3-Show a prediction based upon model and selection: {predictionValue}</p>
                </div>
            )}

            <select
                className="product-dropdown-menu"
                value={selectedProduct}
                onChange={handleProductChange}
            >
                <option value="">Select a product</option>
                {data.map((product) => (
                    <option
                        key={product.idProduct}
                        value={product.idProduct}
                    >
                        {product.name} - Price: ${product.price.toFixed(2)} - Quantity: {product.quantity}
                    </option>
                ))}
            </select>

            {selectedProduct && (
                <div className="details">
                    <h3 className="details-title">Selected Product Details:</h3>
                    <p className="details-item">ID: {selectedProduct}</p>
                    <p className="details-item">Name: {data.find(p => p.idProduct === parseInt(selectedProduct))?.name}</p>
                    <p className="details-item">Price: ${data.find(p => p.idProduct === parseInt(selectedProduct))?.price.toFixed(2)}</p>
                    <p className="details-item">Quantity: {data.find(p => p.idProduct === parseInt(selectedProduct))?.quantity}</p>
                </div>
            )}
        </div>
    );
};

export default Example;