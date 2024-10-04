using System;
using System.Collections.Generic;
using Microsoft.EntityFrameworkCore;

namespace Practice_ML7.Server.Models;

public partial class PrimaryDbContext : DbContext
{
    public PrimaryDbContext()
    {
    }

    public PrimaryDbContext(DbContextOptions<PrimaryDbContext> options)
        : base(options)
    {
    }

    public virtual DbSet<Customer> Customers { get; set; }

    public virtual DbSet<PortFolio> PortFolios { get; set; }

    public virtual DbSet<Product> Products { get; set; }

    public virtual DbSet<TrainingModel> TrainingModels { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
#warning To protect potentially sensitive information in your connection string, you should move it out of source code. You can avoid scaffolding the connection string by using the Name= syntax to read it from configuration - see https://go.microsoft.com/fwlink/?linkid=2131148. For more guidance on storing connection strings, see https://go.microsoft.com/fwlink/?LinkId=723263.
        => optionsBuilder.UseSqlServer("Data Source=cms2024.database.windows.net;Initial Catalog=Primary;User ID=cms;Password=Badpassword1!;Connect Timeout=30;Encrypt=False;Trust Server Certificate=True;Application Intent=ReadWrite;Multi Subnet Failover=False");

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Customer>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK_Customer_id");

            entity.HasOne(d => d.IdCustomerNavigation).WithMany(p => p.Customers).HasConstraintName("FK_Customer_table1_id_Customer");
        });

        modelBuilder.Entity<PortFolio>(entity =>
        {
            entity.HasKey(e => e.IdCustomer).HasName("PK_table1_id_Customer");

            entity.HasOne(d => d.IdProductNavigation).WithMany(p => p.PortFolios).OnDelete(DeleteBehavior.ClientSetNull);
        });

        modelBuilder.Entity<Product>(entity =>
        {
            entity.HasKey(e => e.IdProduct).HasName("PK_table2__id_Product");

            entity.Property(e => e.IdProduct).ValueGeneratedNever();
        });

        modelBuilder.Entity<TrainingModel>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PK_Training_Models_id");
        });

        OnModelCreatingPartial(modelBuilder);
    }

    partial void OnModelCreatingPartial(ModelBuilder modelBuilder);
}
