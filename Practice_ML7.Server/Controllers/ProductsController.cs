using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Practice_ML7.Server.Models;
using Tensorflow;
using static Tensorflow.Binding;

namespace Practice_ML7.Server.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ProductsController : ControllerBase
    {
        private readonly PrimaryDbContext _context;
        private readonly ILogger<ProductsController> _logger;
        public class PredictionData
        {
            public float PredictionDataUpdate { get; set; }
        }
        static ProductsController()
        {
            tf.enable_eager_execution();
            System.Diagnostics.Debug.WriteLine("Eager execution enabled");
        }

        public ProductsController(PrimaryDbContext context, ILogger<ProductsController> logger)
        {
            _context = context;
            _logger = logger;
        }

       

        public class ProductFactory
        {
            public interface IProductFactory
            {
                Task CreateProduct(int id, string name, PrimaryDbContext context);
                Task PhaseOne(int id, string name, PrimaryDbContext context);
                Task PhaseTwo(int id, string name, PrimaryDbContext context);
            }

            public static IProductFactory CreateFactory(int id, string name)
            {
                return new ConcreteProductFactory(id, name);
            }

            private class ConcreteProductFactory : IProductFactory
            {
                private readonly int _id;
                private readonly string _name;

                public ConcreteProductFactory(int id, string name)
                {
                    _id = id;
                    _name = name;
                }

                public async Task CreateProduct(int id, string name, PrimaryDbContext context)
                {
                    System.Diagnostics.Debug.WriteLine($"Initializing CreateProduct method for id: {id}, name: {name}");
                    try
                    {
                        System.Diagnostics.Debug.WriteLine("Initializing TensorFlow operations");
                        tf.enable_eager_execution();
                        System.Diagnostics.Debug.WriteLine("TensorFlow eager execution enabled");

                        System.Diagnostics.Debug.WriteLine("Fetching pricing model from database");
                        var pricingModel = await context.TrainingModels
                            .FirstOrDefaultAsync(m => m.ModelName == "Pricing_Model");

                        PredictionData predictionData = new PredictionData();

                        if (pricingModel != null)
                        {
                            System.Diagnostics.Debug.WriteLine("Existing pricing model found. Initializing fine-tuning process.");

                            using (var memoryStream = new MemoryStream(pricingModel.Data))
                            using (var reader = new BinaryReader(memoryStream))
                            {
                                System.Diagnostics.Debug.WriteLine("Deserializing model parameters");
                                int wLength = reader.ReadInt32();
                                float[] wData = new float[wLength];
                                for (int i = 0; i < wLength; i++)
                                {
                                    wData[i] = reader.ReadSingle();
                                }
                                var W = tf.Variable(wData[0], dtype: TF_DataType.TF_FLOAT);

                                int bLength = reader.ReadInt32();
                                float[] bData = new float[bLength];
                                for (int i = 0; i < bLength; i++)
                                {
                                    bData[i] = reader.ReadSingle();
                                }
                                var b = tf.Variable(bData[0], dtype: TF_DataType.TF_FLOAT);

                                System.Diagnostics.Debug.WriteLine("Model parameters loaded successfully");
                                System.Diagnostics.Debug.WriteLine($"Initialized W value: {W.numpy()}, b value: {b.numpy()}");

                                System.Diagnostics.Debug.WriteLine("Fetching product data for fine-tuning");
                                var productRecord = await context.Products
                                    .Where(p => p.IdProduct == id && p.Name == name)
                                    .FirstOrDefaultAsync();

                                if (productRecord == null)
                                {
                                    System.Diagnostics.Debug.WriteLine($"Product initialization failed. Product with ID {id} and name '{name}' not found.");
                                    throw new ArgumentException($"Product with ID {id} and name '{name}' not found.");
                                }

                                System.Diagnostics.Debug.WriteLine($"Product data fetched. Price: {productRecord.Price}");

                                System.Diagnostics.Debug.WriteLine("Initializing training data");
                                var trainData = tf.constant((float)productRecord.Price, dtype: TF_DataType.TF_FLOAT);
                                System.Diagnostics.Debug.WriteLine($"Training data initialized. Value: {trainData.numpy()}");

                                System.Diagnostics.Debug.WriteLine("Initializing fine-tuning parameters");
                                int epochs = 50;
                                float learningRate = 1e-3f;

                                System.Diagnostics.Debug.WriteLine("Starting fine-tuning process");
                                for (int epoch = 0; epoch < epochs; epoch++)
                                {
                                    try
                                    {
                                        using (var tape = tf.GradientTape())
                                        {
                                            var predictions = tf.add(tf.multiply(trainData, W), b);
                                            var loss = tf.square(tf.subtract(predictions, trainData));

                                            var gradients = tape.gradient(loss, new[] { W, b });

                                            W.assign_sub(tf.multiply(gradients[0], tf.constant(learningRate)));
                                            b.assign_sub(tf.multiply(gradients[1], tf.constant(learningRate)));

                                            if (epoch % 10 == 0)
                                            {
                                                System.Diagnostics.Debug.WriteLine($"Fine-tuning Epoch {epoch}, Loss: {loss.numpy()}");
                                                System.Diagnostics.Debug.WriteLine($"Updated W value: {W.numpy()}, b value: {b.numpy()}");
                                            }
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        System.Diagnostics.Debug.WriteLine($"Error in fine-tuning loop at epoch {epoch}: {ex.Message}");
                                        System.Diagnostics.Debug.WriteLine($"Current W value: {W.numpy()}, b value: {b.numpy()}");
                                        System.Diagnostics.Debug.WriteLine($"Call Stack: {Environment.StackTrace}");
                                        throw new Exception($"Fine-tuning failed at epoch {epoch}", ex);
                                    }
                                }

                                System.Diagnostics.Debug.WriteLine("Fine-tuning completed. Calculating prediction.");
                                var prediction = tf.add(tf.multiply(trainData, W), b);
                                predictionData.PredictionDataUpdate = prediction.numpy().ToArray<float>()[0];

                                System.Diagnostics.Debug.WriteLine($"Prediction calculated: {predictionData.PredictionDataUpdate}");

                                System.Diagnostics.Debug.WriteLine("Initializing model serialization");
                                var updatedModelData = new byte[0];
                                using (var saveStream = new MemoryStream())
                                {
                                    using (var writer = new BinaryWriter(saveStream))
                                    {
                                        writer.Write(1);
                                        writer.Write((float)W.numpy());
                                        writer.Write(1);
                                        writer.Write((float)b.numpy());
                                    }
                                    updatedModelData = saveStream.ToArray();
                                }

                                System.Diagnostics.Debug.WriteLine("Updating pricing model in database");
                                pricingModel.Data = updatedModelData;
                                await context.SaveChangesAsync();
                                System.Diagnostics.Debug.WriteLine("Fine-tuned model saved to TrainingModels table");
                            }
                        }
                        else
                        {
                            System.Diagnostics.Debug.WriteLine("No existing pricing model found. Initializing new model creation.");

                            System.Diagnostics.Debug.WriteLine("Fetching product data");
                            var productRecord = await context.Products
                                .Where(p => p.IdProduct == id && p.Name == name)
                                .FirstOrDefaultAsync();

                            if (productRecord == null)
                            {
                                System.Diagnostics.Debug.WriteLine($"Product initialization failed. Product with ID {id} and name '{name}' not found.");
                                throw new ArgumentException($"Product with ID {id} and name '{name}' not found.");
                            }

                            System.Diagnostics.Debug.WriteLine($"Product data fetched. Price: {productRecord.Price}");

                            System.Diagnostics.Debug.WriteLine("Fetching all products with the same name for training");
                            var productsByName = await context.Products
                               .Where(p => p.Name == name)
                               .ToListAsync();

                            var productsByNamePrices = productsByName.Select(p => (float)p.Price).ToArray();

                            System.Diagnostics.Debug.WriteLine($"Training data initialized. Number of samples: {productsByNamePrices.Length}");
                            System.Diagnostics.Debug.WriteLine($"Price range: {productsByNamePrices.Min()} to {productsByNamePrices.Max()}");

                            System.Diagnostics.Debug.WriteLine("Initializing TensorFlow tensor");
                            Tensor trainData;
                            try
                            {
                                trainData = tf.convert_to_tensor(productsByNamePrices, dtype: TF_DataType.TF_FLOAT);
                                trainData = tf.reshape(trainData, new[] { -1, 1 }); // Reshape to 2D
                                System.Diagnostics.Debug.WriteLine($"Tensor shape initialized: {string.Join(", ", trainData.shape)}");
                            }
                            catch (Exception ex)
                            {
                                System.Diagnostics.Debug.WriteLine($"Tensor initialization failed: {ex.Message}");
                                throw new Exception("Failed to initialize tensor from price data.", ex);
                            }

                            System.Diagnostics.Debug.WriteLine("Initializing model variables");
                            var W = tf.Variable(tf.random.normal(new[] { 1, 1 }));
                            var b = tf.Variable(tf.zeros(new[] { 1 }));

                            System.Diagnostics.Debug.WriteLine($"Initial W shape: {string.Join(", ", W.shape)}, b shape: {string.Join(", ", b.shape)}");

                            System.Diagnostics.Debug.WriteLine("Initializing training parameters");
                            int epochs = 100;
                            float learningRate = 1e-2f;

                            System.Diagnostics.Debug.WriteLine("Starting training process");
                            for (int epoch = 0; epoch < epochs; epoch++)
                            {
                                try
                                {
                                    using (var tape = tf.GradientTape())
                                    {
                                        var predictions = tf.matmul(trainData, W) + b;
                                        var loss = tf.reduce_mean(tf.square(predictions - trainData));

                                        var gradients = tape.gradient(loss, new[] { W, b });

                                        W.assign_sub(gradients[0] * learningRate);
                                        b.assign_sub(gradients[1] * learningRate);

                                        if (epoch % 10 == 0)
                                        {
                                            System.Diagnostics.Debug.WriteLine($"Training Epoch {epoch}, Loss: {loss.numpy()}");
                                        }
                                    }
                                }
                                catch (Exception ex)
                                {
                                    System.Diagnostics.Debug.WriteLine($"Error in training loop at epoch {epoch}: {ex.Message}");
                                    throw new Exception($"Training failed at epoch {epoch}", ex);
                                }
                            }

                            System.Diagnostics.Debug.WriteLine("Training completed. Preparing for prediction.");
                            var inputArray = new float[] { (float)productRecord.Price };
                            var inputTensor = tf.convert_to_tensor(inputArray, dtype: TF_DataType.TF_FLOAT);
                            inputTensor = tf.reshape(inputTensor, new[] { -1, 1 }); // Reshape to 2D

                            System.Diagnostics.Debug.WriteLine("Calculating prediction");
                            var prediction = tf.matmul(inputTensor, W) + b;
                            predictionData.PredictionDataUpdate = prediction.numpy().ToArray<float>()[0];

                            System.Diagnostics.Debug.WriteLine($"Prediction calculated. Original price: {productRecord.Price}, Predicted price: {predictionData.PredictionDataUpdate}");

                            System.Diagnostics.Debug.WriteLine("Initializing model serialization");
                            var modelData = new byte[0];
                            using (var memoryStream = new MemoryStream())
                            {
                                using (var writer = new BinaryWriter(memoryStream))
                                {
                                    var wData = W.numpy().ToArray<float>();
                                    writer.Write(wData.Length);
                                    foreach (var value in wData)
                                    {
                                        writer.Write(value);
                                    }

                                    var bData = b.numpy().ToArray<float>();
                                    writer.Write(bData.Length);
                                    foreach (var value in bData)
                                    {
                                        writer.Write(value);
                                    }
                                }
                                modelData = memoryStream.ToArray();
                            }

                            System.Diagnostics.Debug.WriteLine("Creating new TrainingModel entry");
                            var trainingModel = new TrainingModel
                            {
                                ModelName = "Pricing_Model",
                                Data = modelData
                            };

                            context.TrainingModels.Add(trainingModel);
                            await context.SaveChangesAsync();

                            System.Diagnostics.Debug.WriteLine("New model saved to TrainingModels table");
                        }

                        System.Diagnostics.Debug.WriteLine($"Final PredictionDataUpdate value: {predictionData.PredictionDataUpdate}");
                        System.Diagnostics.Debug.WriteLine("CreateProduct method completed successfully");
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"CreateProduct method failed: {ex.Message}");
                        System.Diagnostics.Debug.WriteLine($"Stack trace: {ex.StackTrace}");
                        throw;
                    }
                }
                public async Task PhaseOne(int id, string name, PrimaryDbContext context)
                {
                    await Task.Run(() => System.Diagnostics.Debug.WriteLine($"PhaseOne: {id}, {name}"));
                }

                public async Task PhaseTwo(int id, string name, PrimaryDbContext context)
                {
                    await Task.Run(() => System.Diagnostics.Debug.WriteLine($"PhaseTwo: {id}, {name}"));
                }
            }
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<Product>>> GetProducts()
        {
            try
            {
                return await _context.Products.ToListAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred while getting products");
                return StatusCode(StatusCodes.Status500InternalServerError, "An error occurred while processing your request.");
            }
        }

        [HttpGet("{id}")]
        public async Task<ActionResult<Product>> GetProduct(int id)
        {
            var product = await _context.Products.FindAsync(id);

            if (product == null)
            {
                return NotFound();
            }

            return product;
        }



        

        [HttpGet("MLActionPrediction")]
        public async Task<ActionResult<float>> GetPrediction()
        {

            return (null);
        }

        [HttpGet("MLAction")]
        public async Task<ActionResult<Product>> MLAction(int id, string name)
        {
            try
            {
                var product = await _context.Products.FindAsync(id);

                if (product == null)
                {
                    return NotFound();
                }

                var factory = ProductFactory.CreateFactory(id, name);

                await factory.CreateProduct(id, name, _context);
                await factory.PhaseOne(id, name, _context);
                await factory.PhaseTwo(id, name, _context);

                return Ok(product);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred during MLAction for product id: {ProductId}", id);
                return StatusCode(StatusCodes.Status500InternalServerError, "An error occurred while processing your request.");
            }
        }
    }
}