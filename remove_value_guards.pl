use strict;
use warnings;

sub indentation {
    my ($line) = @_;
    $line =~ /^(\s*)/;
    return length($1);
}

for my $path (@ARGV) {
    open my $input, "<", $path or die "cannot read $path: $!";
    my @lines = <$input>;
    close $input;

    my @remove;
    for (my $start = 0; $start < @lines; $start++) {
        next unless $lines[$start] =~ /^(\s*)if\b/;
        my $base = length($1);
        my $header_end = $start;
        $header_end++ while (
            $header_end + 1 < @lines
            && $lines[$header_end] !~ /:\s*(?:#.*)?$/
        );
        next if $header_end + 1 >= @lines;

        my $body_start = $header_end + 1;
        $body_start++ while (
            $body_start < @lines && $lines[$body_start] =~ /^\s*$/
        );
        next if $body_start >= @lines;
        my $body_indent = indentation($lines[$body_start]);
        next unless $body_indent > $base;
        next unless $lines[$body_start] =~ /^\s*raise ValueError\b/;

        my $end = $body_start + 1;
        while ($end < @lines) {
            if ($lines[$end] =~ /^\s*$/) {
                $end++;
                next;
            }
            last if indentation($lines[$end]) <= $base;
            last if indentation($lines[$end]) == $body_indent;
            $end++;
        }
        my $next = $end;
        $next++ while ($next < @lines && $lines[$next] =~ /^\s*$/);
        next if (
            $next < @lines
            && indentation($lines[$next]) == $base
            && $lines[$next] =~ /^\s*(?:elif|else)\b/
        );
        push @remove, [$start, $end];
        $start = $end - 1;
    }

    next unless @remove || $ENV{CLEANUP};
    if ($ENV{DRY_RUN}) {
        print "$path: " . scalar(@remove) . "\n";
        next;
    }
    for my $range (reverse @remove) {
        $lines[$range->[0]] =~ /^(\s*)/;
        splice @lines, $range->[0], $range->[1] - $range->[0], "$1pass\n";
    }
    my @drop_pass;
    for (my $index = 0; $index < @lines; $index++) {
        next unless $lines[$index] =~ /^(\s*)pass\s*$/;
        my $pass_indent = length($1);
        my $parent = $index - 1;
        while ($parent >= 0) {
            if (
                $lines[$parent] !~ /^\s*$/
                && indentation($lines[$parent]) < $pass_indent
            ) {
                last;
            }
            $parent--;
        }
        next if $parent < 0;
        my $parent_indent = indentation($lines[$parent]);
        my $end = $index + 1;
        while ($end < @lines) {
            last if (
                $lines[$end] !~ /^\s*$/
                && indentation($lines[$end]) <= $parent_indent
            );
            $end++;
        }
        my @peers = grep {
            $lines[$_] !~ /^\s*$/
            && indentation($lines[$_]) == $pass_indent
            && $lines[$_] !~ /^\s*pass\s*$/
        } ($parent + 1 .. $end - 1);
        my @passes = grep {
            $lines[$_] =~ /^\s*pass\s*$/
        } ($parent + 1 .. $end - 1);
        if (@peers) {
            push @drop_pass, @passes;
        } elsif (@passes > 1) {
            push @drop_pass, @passes[1 .. $#passes];
        }
        $index = $end - 1;
    }
    for my $index (sort { $b <=> $a } keys %{ { map { $_ => 1 } @drop_pass } }) {
        splice @lines, $index, 1;
    }
    open my $output, ">", $path or die "cannot write $path: $!";
    print {$output} @lines;
    close $output;
}
