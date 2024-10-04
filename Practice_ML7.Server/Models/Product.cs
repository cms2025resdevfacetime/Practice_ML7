using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace Practice_ML7.Server.Models;

[Table("Product")]
public partial class Product
{
    [Key]
    [Column("id_Product")]
    public int IdProduct { get; set; }

    [StringLength(50)]
    [Unicode(false)]
    public string? Name { get; set; }

    [Column(TypeName = "money")]
    public decimal? Price { get; set; }

    public int? Quantity { get; set; }

    [InverseProperty("IdProductNavigation")]
    public virtual ICollection<PortFolio> PortFolios { get; set; } = new List<PortFolio>();
}
