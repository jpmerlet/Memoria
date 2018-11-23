#!/usr/bin/perl

if ($#ARGV != 1) { die "Usage: convertAMPL_upit.pl <file.upit> <file.prec> ";}

#open file
$filename = $ARGV[0];

printf STDERR  "Reading file $filename\n";
open (FH, "<", "$filename") || die "Can't open $filename: $!";
substr $filename, index($filename, '.upit'), 6, '';
my $output_filename = join "", $filename,'_upit_input','.dat';
open(my $fh, '>', "$output_filename");
while(<FH>)
{
	chomp $_;
	my @col = split(/:/,$_);
	if ($col[0] eq 'NBLOCKS') { 
		printf $fh "param NBLOCKS := $col[1];\n";
		$nb = $col[1];
	}
		if ($col[0] eq 'OBJECTIVE_FUNCTION')
		{
			print $fh "param: OBJECTIVE_FUNCTION_UPIT :=";
			for($i = 0 ; $i < $nb ; $i++)
			{
				$_ = <FH>;
				chomp $_;
				my @col = split(/ /,$_);
				print $fh "\n$col[0] $col[1]";
			}
			printf $fh ";"
		}

}
close $fh;
close(FH);

$filename = $ARGV[1];

printf STDERR  "Reading file $filename\n";
open (FH, "<", "$filename") || die "Can't open $filename: $!";
substr $filename, index($filename, '.prec'), 6, '';
my $output_filename = join "", $filename,'_prec','.dat';
open(my $fh, '>', "$output_filename");


for ($i = 0 ; $i < $nb ; $i++)
{
	$_ = <FH>;
	chomp $_;
	my @col = split(/ /,$_);
	if ($col[0] != $i) { die "ERROR: row $i does not coincide (block $col[0]\n";}
	printf $fh "set PREC[$col[0]] :=";
	for ($j = 0 ; $j < $col[1] ; $j++)
	{
		printf $fh " $col[2+$j]";
	}
	printf $fh ";\n";
}
close(fh);
close(FH);