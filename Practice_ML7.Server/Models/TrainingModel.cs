using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace Practice_ML7.Server.Models;

[Table("Training_Models")]
public partial class TrainingModel
{
    [Key]
    [Column("id")]
    public int Id { get; set; }

    [Column("Model_Name")]
    [StringLength(50)]
    [Unicode(false)]
    public string? ModelName { get; set; }

    public byte[]? Data { get; set; }
}
