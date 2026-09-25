"""Create a demo user and a work-permit case with fictional documents."""

from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand

from cases.services.cases import add_document, create_case

SAMPLES = Path(settings.BASE_DIR) / "sample_documents"
USERNAME, PASSWORD = "reviewer", "demo-password"


class Command(BaseCommand):
    help = "Seed a demo reviewer account and one case with three fictional documents."

    def handle(self, *args, **options):
        User = get_user_model()
        user, created = User.objects.get_or_create(username=USERNAME, defaults={"is_staff": True})
        if created:
            user.set_password(PASSWORD)
            user.save()

        case = create_case(
            title="Work permit - Amara Okafor", case_type="work_permit", applicant_name="Amara Okafor", user=user
        )
        for path in sorted(SAMPLES.glob("*.txt")):
            data = path.read_bytes()
            upload = SimpleUploadedFile(path.name, data, content_type="text/plain")
            add_document(case=case, upload=upload, text=data.decode("utf-8"), user=user)

        self.stdout.write(self.style.SUCCESS(f"Created case #{case.pk} with {case.documents.count()} documents."))
        self.stdout.write(f"Log in as {USERNAME} / {PASSWORD}")
