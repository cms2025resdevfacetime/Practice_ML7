using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace Practice_ML7.Server.Models;

[Table("PortFolio")]
public partial class PortFolio
{
    [Key]
    [Column("id_Customer")]
    public int IdCustomer { get; set; }

    [Column("id_Product")]
    public int IdProduct { get; set; }

    [InverseProperty("IdCustomerNavigation")]
    public virtual ICollection<Customer> Customers { get; set; } = new List<Customer>();

    [ForeignKey("IdProduct")]
    [InverseProperty("PortFolios")]
    public virtual Product IdProductNavigation { get; set; } = null!;
}
