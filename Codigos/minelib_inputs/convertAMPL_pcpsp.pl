#!/usr/bin/perl

if ($#ARGV != 1) { die "Usage: convertAMPL_pcpsp.pl <file.pcpsp> <file.prec> ";}

#open file
$filename = $ARGV[0];

printf "data;\n";

printf STDERR  "Reading file $filename\n";
open (FH, "<", "$filename") || die "Can't open $filename: $!";
substr $filename, index($filename, '.pcpsp'), 6, '';
my $output_filename = join "", $filename,'_pcpsp_input','.dat';
open(my $fh, '>', "$output_filename");
while(<FH>)
{
	chomp $_;
	my @col = split(/:/,$_);
	if ($col[0] eq 'NBLOCKS') { 
#		printf $fh "param NBLOCKS := $col[1];\n";
		$nb = $col[1];
	}
	if ($col[0] eq 'NPERIODS') {
	 	printf $fh "param NPERIODS := $col[1];\n";
 		$nt = $col[1];
	}
	if ($col[0] eq 'NDESTINATIONS') {
	 	printf $fh "param NDESTINATIONS := $col[1];\n";
 		$nd = $col[1];
	}
	if ($col[0] eq 'NRESOURCE_SIDE_CONSTRAINTS') { 
		printf $fh "param NRESOURCE_SIDE_CONSTRAINTS := $col[1];\n";
 		$nr = $col[1];
	}
	if ($col[0] eq 'DISCOUNT_RATE') {
	 	printf $fh "param DISCOUNT_RATE := $col[1];\n"; 
	}
	if ($col[0] eq 'RESOURCE_CONSTRAINT_LIMITS')
	{
		for($i = 0 ; $i < $nr*$nt ; $i++)
		{
				$_ = <FH>;
				chomp $_;
				my @col = split(/ /,$_);
				if($col[2] eq 'L')
				{
					$ub[$col[0]][$col[1]] = $col[3];
					$lb[$col[0]][$col[1]] = -infinity;
				}
				elsif ($col[2] eq 'G')
				{
					$ub[$col[0]][$col[1]] = infinity;
					$lb[$col[0]][$col[1]] = $col[3];
				}
				elsif ($col[2] eq 'I')
				{
					$ub[$col[0]][$col[1]] = $col[4];
					$lb[$col[0]][$col[1]] = $col[3];
				}
			}

			printf $fh "param RESOURCE_CONSTRAINT_UB_LIMITS :";
			for ($t = 0 ; $t < $nt ; $t++) {printf $fh " $t";}
			printf $fh ":= ";
			for ($r = 0 ; $r < $nr ; $r++)
			{
				printf $fh "\n$r";
				for ($t = 0 ; $t < $nt ; $t++) {printf $fh " $ub[$r][$t]";}
			}
			printf $fh ";\n";

			printf $fh "param RESOURCE_CONSTRAINT_LB_LIMITS :";
			for ($t = 0 ; $t < $nt ; $t++) {printf $fh " $t";}
			printf $fh ":= ";
			for ($r = 0 ; $r < $nr ; $r++)
			{
				printf $fh "\n$r";
				for ($t = 0 ; $t < $nt ; $t++) {printf $fh " $lb[$r][$t]";}
			}
			printf $fh ";\n";
		}

		if ($col[0] eq 'OBJECTIVE_FUNCTION')
		{
			print $fh "param OBJECTIVE_FUNCTION_PCPSP :";
			for($d = 0 ; $d < $nd ; $d++)
			{
				printf $fh (" $d");
			}
			printf $fh (" :=");
			for($i = 0 ; $i < $nb ; $i++)
			{
				$_ = <FH>;
				chomp $_;
				my @col = split(/ /,$_);
				print $fh "\n$col[0]";
				for($d = 0 ; $d < $nd ; $d++)
				{
					printf $fh " $col[$d+1]";
				}
			}
			printf $fh ";\n"
		}

		if ($col[0] eq 'RESOURCE_CONSTRAINT_COEFFICIENTS')
		{
			print $fh "param: RESOURCE_CONSTRAINT_COEFFICIENTS :=";
			while(<FH>)
			{
				chomp $_;
				my @col = split(/ /,$_);
				if ($col[0] eq "EOF") { break; }
				else {
					printf $fh "\n$col[0] $col[2] $col[1] $col[3]";
				}
			}
			printf $fh ";\n";
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