using System;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using System.Linq;
using Microsoft.Identity.Client;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.HttpLogging;
using Microsoft.Extensions.DependencyInjection;
using Practice_ML7.Server.Models;
//Modification 10.19.2024
namespace Practice_ML7.Server
{
    public class Program
    {
        // Import the LoadLibrary function from kernel32.dll
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr LoadLibrary(string lpFileName);

        public static void Main(string[] args)
        {


            // Create a new WebApplicationBuilder
            var builder = WebApplication.CreateBuilder(args);

            // Configure Kestrel web server
            builder.WebHost.ConfigureKestrel(serverOptions =>
            {
                var configuration = builder.Configuration;
                serverOptions.ListenAnyIP(int.Parse(configuration["Kestrel:Endpoints:Http:Url"].Split(':').Last()));
                serverOptions.ListenAnyIP(int.Parse(configuration["Kestrel:Endpoints:Https:Url"].Split(':').Last()), listenOptions =>
                {
                    listenOptions.UseHttps();
                });
            });

            // Add HTTP logging to the services
            builder.Services.AddHttpLogging(logging =>
            {
                logging.LoggingFields = HttpLoggingFields.All;
                logging.RequestHeaders.Add("Origin");
                logging.ResponseHeaders.Add("Access-Control-Allow-Origin");
            });

            // Add memory cache for server-side caching (useful for ML learning models)
            builder.Services.AddMemoryCache();

            // Register the database context
            builder.Services.AddDbContext<PrimaryDbContext>(options =>
                options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

            // Configure CORS policy
            builder.Services.AddCors(options =>
            {
                options.AddDefaultPolicy(builder =>
                {
                    builder.SetIsOriginAllowed(_ => true)
                           .AllowAnyMethod()
                           .AllowAnyHeader()
                           .AllowCredentials();
                });
            });

            // Add other services
            builder.Services.AddHttpClient();
            builder.Services.AddControllers();
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            // Configure the HTTP request pipeline
            if (app.Environment.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpLogging();
            app.UseCors();
            app.UseDefaultFiles();
            app.UseStaticFiles();
            app.UseAuthorization();
            app.MapControllers();
            app.MapFallbackToFile("/index.html");

            app.Run();
        }

    }
}
